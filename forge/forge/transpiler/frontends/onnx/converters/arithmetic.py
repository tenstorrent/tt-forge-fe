# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ONNX Arithmetic operation converters.

This module provides converters for ONNX arithmetic operations:
- Add, Sub, Mul, Div: Element-wise binary operations with broadcasting support
- MatMul: Matrix multiplication operation

Key features:
- Handles opset version differences in broadcasting behavior (v1-6 vs v7+)
- Validates shape compatibility based on opset version
- Supports PyTorch-style multidirectional broadcasting (opset 7+)
- Handles limited broadcasting with axis attribute (opset 1-6)
"""
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from onnx import NodeProto
from forge.transpiler.core.types import TensorInfo
from forge.transpiler.operations.arithmetic import AddNode, SubNode, MulNode, DivNode, MatMulNode
from forge.transpiler.frontends.onnx.converters.base import OnnxOpConverter
from forge.transpiler.frontends.onnx.utils.io_builder import build_input_output_dicts
from forge.transpiler.frontends.onnx.utils.broadcasting import (
    are_shapes_compatible_for_broadcasting,
    compute_broadcasted_shape,
)


def _are_shapes_equal(shape_a: Tuple, shape_b: Tuple) -> bool:
    """
    Check if two shapes are exactly equal.

    Args:
        shape_a: First shape tuple
        shape_b: Second shape tuple

    Returns:
        True if shapes are identical, False otherwise
    """
    return shape_a == shape_b


def _validate_limited_broadcasting(shape_a: Tuple, shape_b: Tuple, axis: Optional[int], op_type: str) -> None:
    """
    Validate broadcasting for OPSET 1-6 (limited broadcasting with axis attribute).

    In OPSET 1-6, broadcasting is limited:
    - B's shape must match a contiguous subset of A's shape
    - If axis is specified, B aligns starting at that axis
    - If axis is not specified, suffix matching is used (B aligns from right)

    Args:
        shape_a: Shape of tensor A (left operand)
        shape_b: Shape of tensor B (right operand)
        axis: Optional axis attribute (None means suffix matching)
        op_type: Operation type for error messages

    Raises:
        ValueError: If shapes are not compatible for limited broadcasting
    """
    shape_a_list = list(shape_a)
    shape_b_list = list(shape_b)

    if axis is None:
        # Suffix matching: B must match the suffix of A (aligned from right)
        # This is the default behavior when axis is not specified
        if len(shape_b_list) > len(shape_a_list):
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET 1-6, suffix matching): "
                f"Shape B {shape_b} has more dimensions than shape A {shape_a}. "
                f"B must match the suffix of A."
            )

        # Check compatibility from right to left (suffix matching)
        for i in range(len(shape_b_list)):
            dim_a = shape_a_list[-(i + 1)]
            dim_b = shape_b_list[-(i + 1)]

            if dim_a != dim_b:
                raise ValueError(
                    f"Broadcasting error in {op_type} (OPSET 1-6, suffix matching): "
                    f"Shapes {shape_a} and {shape_b} are not compatible. "
                    f"Dimension mismatch at position {len(shape_a_list) - i - 1}: {dim_a} vs {dim_b}"
                )
    else:
        # Axis-specified matching: B aligns starting at the specified axis in A
        # This allows B to be placed at a specific position in A's shape
        if axis < 0 or axis >= len(shape_a_list):
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET 1-6, axis={axis}): "
                f"Axis {axis} is out of range for shape A {shape_a} "
                f"(valid range: 0 to {len(shape_a_list) - 1})"
            )

        # B's dimensions must fit within A's dimensions starting at axis
        if len(shape_b_list) > len(shape_a_list) - axis:
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET 1-6, axis={axis}): "
                f"Shape B {shape_b} has {len(shape_b_list)} dimensions, but only "
                f"{len(shape_a_list) - axis} dimensions available starting at axis {axis} "
                f"in shape A {shape_a}"
            )

        # Check dimension compatibility starting at the specified axis
        for i in range(len(shape_b_list)):
            dim_a = shape_a_list[axis + i]
            dim_b = shape_b_list[i]

            if dim_a != dim_b:
                raise ValueError(
                    f"Broadcasting error in {op_type} (OPSET 1-6, axis={axis}): "
                    f"Shapes {shape_a} and {shape_b} are not compatible. "
                    f"Dimension mismatch at axis {axis + i}: {dim_a} vs {dim_b}"
                )


def _validate_broadcast_attributes(
    op_type: str, attrs: Dict[str, Any], input_tensors: OrderedDict[str, TensorInfo], opset: int
) -> None:
    """
    Validate broadcast and axis attributes based on opset version.

    This function only validates - it does not return processed attributes.
    Raises ValueError if validation fails.

    Args:
        op_type: Operation type (Add, Sub, Mul, Div)
        attrs: Extracted attributes dictionary
        input_tensors: Dictionary of input tensor information
        opset: Opset version

    Raises:
        ValueError: If shapes are incompatible for broadcasting
    """
    # Extract broadcast and axis attributes
    broadcast = attrs.get("broadcast", 0)
    axis = attrs.get("axis", None)

    # Get input shapes and validate
    if len(input_tensors) < 2:
        raise ValueError(f"{op_type} node: Expected 2 inputs, got {len(input_tensors)}")

    input_names = list(input_tensors.keys())
    tensor_a = input_tensors[input_names[0]]
    tensor_b = input_tensors[input_names[1]]

    shape_a = tensor_a.shape
    shape_b = tensor_b.shape

    if shape_a is None or shape_b is None:
        logger.warning(
            f"{op_type} node: Cannot validate broadcasting - one or both shapes are unknown. "
            f"Shape A: {shape_a}, Shape B: {shape_b}"
        )
        return  # Skip validation if shapes are unknown

    shapes_match = _are_shapes_equal(shape_a, shape_b)
    shapes_compatible_multidir = are_shapes_compatible_for_broadcasting(shape_a, shape_b)

    # Handle broadcasting validation based on opset version
    if opset <= 6:
        # OPSET 1-6: Limited broadcasting, requires explicit broadcast=1 attribute
        # Broadcasting is opt-in, not automatic
        if not shapes_match:
            if broadcast == 0:
                raise ValueError(
                    f"Broadcasting error in {op_type} (OPSET {opset}): "
                    f"Shapes {shape_a} and {shape_b} don't match and broadcast=0. "
                    f"Set broadcast=1 to enable broadcasting."
                )
            else:
                # broadcast=1 is set, validate limited broadcasting rules
                # Limited broadcasting: B must match a contiguous subset of A's shape
                _validate_limited_broadcasting(shape_a, shape_b, axis, op_type)
                if axis is not None:
                    logger.debug(f"{op_type} node: Using axis={axis} for broadcasting (OPSET {opset})")

    else:
        # OPSET 7+: Multidirectional broadcasting always enabled (NumPy/PyTorch style)
        # The broadcast and axis attributes were removed in opset 7
        if broadcast != 0 or axis is not None:
            logger.warning(
                f"{op_type} node: 'broadcast' and 'axis' attributes are not supported "
                f"in OPSET {opset} (removed in OPSET 7+). These attributes will be ignored. "
                f"Multidirectional broadcasting is always enabled."
            )

        # Validate shapes are compatible for multidirectional broadcasting
        # In multidirectional broadcasting, dimensions align from the right
        # and are compatible if equal or one is 1
        if not shapes_match and not shapes_compatible_multidir:
            raise ValueError(
                f"Broadcasting error in {op_type} (OPSET {opset}): "
                f"Shapes {shape_a} and {shape_b} are not compatible for "
                f"multidirectional broadcasting. "
                f"Two dimensions are compatible if they are equal OR one is 1."
            )


class BinaryArithmeticConverter(OnnxOpConverter):
    """
    Unified converter for binary arithmetic operations: Add, Sub, Mul, Div.

    This converter handles all four operations (Add, Sub, Mul, Div) using a single
    implementation, while maintaining separate operator nodes for each operation type.
    The operator nodes use PyTorch API (torch.add, torch.sub, torch.mul, torch.div).

    Broadcasting Handling:
    - OPSET 1-6: Validates `broadcast=1` attribute and handles `axis` attribute
    - OPSET 7+: Multidirectional broadcasting always enabled (attributes ignored)
    - PyTorch operations handle broadcasting automatically
    """

    # Mapping from ONNX op type to corresponding node class
    _OP_NODE_MAP = {
        "Add": AddNode,
        "Sub": SubNode,
        "Mul": MulNode,
        "Div": DivNode,
    }

    @classmethod
    def convert(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset: int = 1,
    ) -> List:
        """
        Convert binary arithmetic operations (Add, Sub, Mul, Div).

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (may include broadcast, axis)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer
            opset: Opset version (default: 1)

        Returns:
            List containing a single node instance (AddNode, SubNode, MulNode, or DivNode)
        """
        op_type = node_proto.op_type

        # Get the appropriate node class for this operation
        node_class = cls._OP_NODE_MAP.get(op_type)
        if node_class is None:
            raise ValueError(
                f"Unsupported binary arithmetic operation: {op_type}. "
                f"Supported operations: {list(cls._OP_NODE_MAP.keys())}"
            )

        # Validate broadcast/axis attributes based on opset version
        # This ensures shapes are compatible and raises errors if not
        # Note: PyTorch operations handle broadcasting automatically, but we validate
        # to catch errors early and provide better error messages
        _validate_broadcast_attributes(op_type=op_type, attrs=attrs, input_tensors=input_tensors, opset=opset)

        # Generate node name if not provided
        node_name = node_proto.name if node_proto.name else f"{op_type}_{node_index}"

        # Compute output shape if not already set (for opset 7+ with multidirectional broadcasting)
        if opset >= 7 and len(input_tensors) >= 2:
            input_names = list(input_tensors.keys())
            tensor_a = input_tensors[input_names[0]]
            tensor_b = input_tensors[input_names[1]]

            output_shape = compute_broadcasted_shape(tensor_a.shape, tensor_b.shape)
            if output_shape is not None and len(output_tensors) > 0:
                output_name = list(output_tensors.keys())[0]
                output_tensors[output_name] = TensorInfo(
                    name=output_name,
                    shape=output_shape,
                    onnx_dtype=tensor_a.onnx_dtype,  # Output dtype matches input dtypes (validated to match)
                )

        # Build OrderedDict for inputs and outputs
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        # Create and return the appropriate node
        # The node will use PyTorch operations (torch.add, torch.sub, etc.)
        # which handle broadcasting automatically
        return [node_class.create(name=node_name, inputs=input_dict, outputs=output_dict)]


def _validate_matmul_shapes(
    node_name: str, shape_a: Optional[Tuple], shape_b: Optional[Tuple], opset: int
) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple]]:
    """
    Validate shapes for MatMul operation and compute output shape.

    MatMul performs matrix multiplication: Y = A @ B
    - For 2D: A [M, K] @ B [K, N] -> Y [M, N]
    - For N-dimensional: A [..., M, K] @ B [..., K, N] -> Y [..., M, N]
    - Batch dimensions are broadcastable

    Args:
        node_name: Name of the MatMul node (for error messages)
        shape_a: Shape of input tensor A
        shape_b: Shape of input tensor B
        opset: ONNX opset version

    Returns:
        Tuple of (shape_a, shape_b, output_shape) after validation

    Raises:
        ValueError: If shapes are incompatible for matrix multiplication
    """
    if shape_a is None or shape_b is None:
        logger.warning(
            f"MatMul node {node_name}: Cannot fully validate shapes - "
            f"one or both shapes are unknown. Shape A: {shape_a}, Shape B: {shape_b}"
        )
        return shape_a, shape_b, None

    # Validate minimum dimensions
    if len(shape_a) < 1:
        raise ValueError(f"MatMul node {node_name}: Input A must have at least 1 dimension, got shape {shape_a}")
    if len(shape_b) < 1:
        raise ValueError(f"MatMul node {node_name}: Input B must have at least 1 dimension, got shape {shape_b}")

    # Handle 1D inputs: treat as 2D with leading dimension 1
    # ONNX MatMul requires at least 2D, but PyTorch matmul handles 1D
    # We'll validate as if they're 2D for consistency with ONNX spec
    if len(shape_a) == 1:
        # 1D A: [K] -> treat as [1, K] for validation
        shape_a_2d = (1,) + shape_a
        logger.debug(f"MatMul node {node_name}: Treating 1D input A {shape_a} as 2D {shape_a_2d}")
    else:
        shape_a_2d = shape_a

    if len(shape_b) == 1:
        # 1D B: [K] -> treat as [K, 1] for validation
        shape_b_2d = shape_b + (1,)
        logger.debug(f"MatMul node {node_name}: Treating 1D input B {shape_b} as 2D {shape_b_2d}")
    else:
        shape_b_2d = shape_b

    # Now both shapes are at least 2D
    # Validate matrix multiplication compatibility: A.shape[-1] == B.shape[-2]
    if shape_a_2d[-1] != shape_b_2d[-2]:
        raise ValueError(
            f"MatMul node {node_name}: Incompatible shapes for matrix multiplication. "
            f"A.shape[-1] ({shape_a_2d[-1]}) must equal B.shape[-2] ({shape_b_2d[-2]}). "
            f"A shape: {shape_a} (treated as {shape_a_2d}), B shape: {shape_b} (treated as {shape_b_2d})"
        )

    # Determine if this is batched (N-dimensional) or standard (2D) matrix multiplication
    is_batched = len(shape_a_2d) > 2 or len(shape_b_2d) > 2

    if is_batched:
        # N-dimensional case: validate batch dimensions are broadcastable
        batch_dims_a = shape_a_2d[:-2]
        batch_dims_b = shape_b_2d[:-2]

        # Compute broadcasted batch dimensions
        # Align from right and compute broadcasted shape
        max_batch_len = max(len(batch_dims_a), len(batch_dims_b))
        broadcasted_batch_dims = []

        for i in range(max_batch_len):
            # Get dimensions from right to left
            idx_a = len(batch_dims_a) - max_batch_len + i if i < len(batch_dims_a) else None
            idx_b = len(batch_dims_b) - max_batch_len + i if i < len(batch_dims_b) else None

            dim_a = batch_dims_a[idx_a] if idx_a is not None and idx_a >= 0 else 1
            dim_b = batch_dims_b[idx_b] if idx_b is not None and idx_b >= 0 else 1

            # Broadcasted dimension is max of the two (or 1 if one is missing)
            if dim_a == dim_b:
                broadcasted_batch_dims.append(dim_a)
            elif dim_a == 1:
                broadcasted_batch_dims.append(dim_b)
            elif dim_b == 1:
                broadcasted_batch_dims.append(dim_a)
            else:
                # This should have been caught by broadcasting check, but validate anyway
                raise ValueError(
                    f"MatMul node {node_name}: Batch dimensions are not broadcastable. "
                    f"A batch dims: {batch_dims_a}, B batch dims: {batch_dims_b}. "
                    f"At position {i}: {dim_a} vs {dim_b} (both must be equal or one must be 1)"
                )

        # Validate batch dimensions are compatible for broadcasting
        if not are_shapes_compatible_for_broadcasting(batch_dims_a, batch_dims_b):
            raise ValueError(
                f"MatMul node {node_name}: Batch dimensions are not broadcastable. "
                f"A batch dims: {batch_dims_a}, B batch dims: {batch_dims_b}. "
                f"Batch dimensions must be compatible for NumPy/PyTorch-style broadcasting "
                f"(aligned from right, compatible if equal or one is 1)"
            )

        # Compute output shape: [broadcasted_batch_dims..., M, N]
        # M = A.shape[-2], N = B.shape[-1]
        M = shape_a_2d[-2]
        N = shape_b_2d[-1]
        output_shape = tuple(broadcasted_batch_dims) + (M, N)

        logger.debug(
            f"MatMul node {node_name}: Batched matrix multiplication. "
            f"A: {shape_a} -> {shape_a_2d}, B: {shape_b} -> {shape_b_2d}, "
            f"Batch dims: {batch_dims_a} x {batch_dims_b} -> {broadcasted_batch_dims}, "
            f"Output: {output_shape}"
        )
    else:
        # Standard 2D matrix multiplication
        M = shape_a_2d[-2]
        N = shape_b_2d[-1]
        output_shape = (M, N)

        logger.debug(
            f"MatMul node {node_name}: Standard 2D matrix multiplication. "
            f"A: {shape_a} -> {shape_a_2d} [M={M}, K={shape_a_2d[-1]}], "
            f"B: {shape_b} -> {shape_b_2d} [K={shape_b_2d[-2]}, N={N}], "
            f"Output: {output_shape} [M={M}, N={N}]"
        )

    return shape_a, shape_b, output_shape


class MatMulConverter(OnnxOpConverter):
    """
    Converter for ONNX MatMul (Matrix Multiplication) operation.

    MatMul performs matrix product that behaves like numpy.matmul.
    It computes Y = A @ B where A and B are N-dimensional matrices,
    with the last two dimensions being treated as matrices and all
    preceding dimensions as batch dimensions.

    Key features:
    - No attributes: MatMul has no configurable attributes
    - Standard 2D: Handles standard matrix multiplication [M, K] @ [K, N] -> [M, N]
    - N-dimensional: Handles batched matrix multiplication [..., M, K] @ [..., K, N] -> [..., M, N]
    - Broadcasting: Automatically handles broadcasting for batch dimensions
    - Shape validation: Comprehensive validation of matrix dimensions and batch broadcasting
    - All opset versions: Behavior is consistent across all opset versions (1, 9, 13)
    """

    @classmethod
    def convert(
        cls,
        node_proto: NodeProto,
        input_tensors: OrderedDict[str, TensorInfo],
        output_tensors: OrderedDict[str, TensorInfo],
        attrs: Dict[str, Any],
        node_index: int,
        graph_proto=None,
        opset: int = 1,
    ) -> List:
        """
        Convert ONNX MatMul operation to MatMulNode.

        Supports both standard 2D matrix multiplication and N-dimensional batched
        matrix multiplication with automatic broadcasting of batch dimensions.

        Args:
            node_proto: ONNX node protocol buffer
            input_tensors: Dictionary of input tensor information
            output_tensors: Dictionary of output tensor information
            attrs: Extracted attributes (MatMul has no attributes, but kept for consistency)
            node_index: Index of the node in the graph
            graph_proto: Optional graph protocol buffer
            opset: Opset version (default: 1)

        Returns:
            List containing a single MatMulNode instance

        Raises:
            ValueError: If inputs are invalid or shapes are incompatible
        """
        # Validate inputs
        if len(node_proto.input) != 2:
            raise ValueError(
                f"MatMul node {node_proto.name or f'MatMul_{node_index}'}: "
                f"Expected 2 inputs, got {len(node_proto.input)}"
            )

        # Get input tensor info
        input_a_name = node_proto.input[0]
        input_b_name = node_proto.input[1]

        if input_a_name not in input_tensors:
            raise ValueError(
                f"MatMul node {node_proto.name or f'MatMul_{node_index}'}: "
                f"Input A '{input_a_name}' not found in input_tensors"
            )
        if input_b_name not in input_tensors:
            raise ValueError(
                f"MatMul node {node_proto.name or f'MatMul_{node_index}'}: "
                f"Input B '{input_b_name}' not found in input_tensors"
            )

        tensor_a = input_tensors[input_a_name]
        tensor_b = input_tensors[input_b_name]

        # Validate shapes and compute output shape
        node_name = node_proto.name or f"MatMul_{node_index}"
        shape_a, shape_b, output_shape = _validate_matmul_shapes(node_name, tensor_a.shape, tensor_b.shape, opset)

        # Update output tensor shape if we computed it
        if output_shape is not None and len(output_tensors) > 0:
            output_name = list(output_tensors.keys())[0]
            # Update the output tensor info with computed shape
            output_tensors[output_name] = TensorInfo(
                name=output_name,
                shape=output_shape,
                onnx_dtype=tensor_a.onnx_dtype,  # Output dtype matches input dtype
            )

        # Build OrderedDict for inputs and outputs
        input_dict, output_dict = build_input_output_dicts(node_proto, input_tensors, output_tensors)

        # Create and return MatMulNode
        # MatMulNode uses torch.matmul which handles:
        # - Standard 2D matrix multiplication
        # - N-dimensional batched matrix multiplication
        # - Broadcasting of batch dimensions automatically
        return [MatMulNode.create(name=node_name, inputs=input_dict, outputs=output_dict)]
