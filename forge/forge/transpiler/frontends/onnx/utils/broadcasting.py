# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Broadcasting utilities for ONNX operation converters.

This module provides shared functions for validating and computing broadcasted shapes
following NumPy/PyTorch-style multidirectional broadcasting rules.
"""
from typing import Optional, Tuple


def are_shapes_compatible_for_broadcasting(shape_a: Tuple, shape_b: Tuple) -> bool:
    """
    Check if two shapes are compatible for NumPy-style broadcasting (OPSET 7+).

    Two shapes are compatible for multidirectional broadcasting if:
    - They are equal, OR
    - For each dimension (aligned from right), they are equal OR one is 1 OR one is missing

    This implements PyTorch/NumPy-style broadcasting where dimensions are aligned
    from the right, and missing dimensions are treated as 1.

    Args:
        shape_a: First shape tuple
        shape_b: Second shape tuple

    Returns:
        True if shapes are compatible for broadcasting, False otherwise
    """
    if shape_a is None or shape_b is None:
        return False

    if shape_a == shape_b:
        return True

    # Align shapes from the right (broadcasting aligns trailing dimensions)
    len_a, len_b = len(shape_a), len(shape_b)
    max_len = max(len_a, len_b)

    # Check compatibility dimension by dimension, starting from the right
    for i in range(max_len):
        # Get dimensions from right to left (trailing dimensions first)
        dim_a = shape_a[-(i + 1)] if i < len_a else 1
        dim_b = shape_b[-(i + 1)] if i < len_b else 1

        # Dimensions are compatible if:
        # - They are equal, OR
        # - One of them is 1 (can be broadcast)
        # Missing dimensions (when one shape is shorter) are treated as 1
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False

    return True


def compute_broadcasted_shape(shape_a: Tuple, shape_b: Tuple) -> Optional[Tuple]:
    """
    Compute the broadcasted shape of two input shapes following NumPy-style broadcasting rules.

    Broadcasting rules:
    1. Shapes are compared from right to left
    2. Two dimensions are compatible if:
       - They are equal, OR
       - One of them is 1, OR
       - One of them doesn't exist (missing dimension is treated as 1)
    3. The output shape has the maximum size in each dimension

    Args:
        shape_a: First shape tuple
        shape_b: Second shape tuple

    Returns:
        Broadcasted shape tuple, or None if shapes are incompatible or unknown
    """
    if shape_a is None or shape_b is None:
        return None

    if shape_a == shape_b:
        return shape_a

    # Convert to lists for easier manipulation
    shape_a_list = list(shape_a)
    shape_b_list = list(shape_b)

    # Pad shorter shape with 1s on the left (missing dimensions treated as 1)
    max_len = max(len(shape_a_list), len(shape_b_list))
    shape_a_padded = [1] * (max_len - len(shape_a_list)) + shape_a_list
    shape_b_padded = [1] * (max_len - len(shape_b_list)) + shape_b_list

    # Compute broadcasted shape (maximum size in each dimension)
    broadcasted_shape = []
    for i in range(max_len):
        dim_a = shape_a_padded[i]
        dim_b = shape_b_padded[i]

        # Check compatibility
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return None  # Incompatible shapes

        # Output dimension is the maximum of the two
        broadcasted_dim = max(dim_a, dim_b)
        broadcasted_shape.append(broadcasted_dim)

    return tuple(broadcasted_shape)


def compute_broadcasted_shape_multi(*shapes: Tuple) -> Optional[Tuple]:
    """
    Compute the broadcasted shape of multiple input shapes following NumPy-style broadcasting rules.

    This function broadcasts multiple shapes together by iteratively broadcasting pairs.

    Args:
        *shapes: Variable number of shape tuples to broadcast

    Returns:
        Broadcasted shape tuple, or None if shapes are incompatible or unknown
    """
    if len(shapes) == 0:
        return None

    if len(shapes) == 1:
        return shapes[0]

    # Start with the first shape
    result_shape = shapes[0]

    # Iteratively broadcast with each subsequent shape
    for shape in shapes[1:]:
        result_shape = compute_broadcasted_shape(result_shape, shape)
        if result_shape is None:
            return None  # Incompatible shapes

    return result_shape
