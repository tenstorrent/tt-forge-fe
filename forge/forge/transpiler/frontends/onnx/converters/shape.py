"""
ONNX Shape operation converters (Transpose, Cast).
"""
from typing import List, Dict, Any, Optional
from onnx import NodeProto
import onnx
from ....ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from ....ir.operations.shape import TransposeNode
from ....ir.operations.other import CastNode
from .base import OnnxOpConverter
from .validation import validate_constant_input, validate_attributes


class TransposeConverter(OnnxOpConverter):
    """Converter for ONNX Transpose operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Transpose opset v1+: No version differences.
        
        ONNX Transpose uses perm=[0,2,1,3] which can swap multiple dimensions.
        PyTorch transpose only swaps two dimensions, so we decompose the permutation
        into a series of two-dimension swaps.
        """
        import logging
        logger = logging.getLogger("ForgeTranspiler")
        
        nodes = []
        perm = attrs.get('perm', None)
        
        if perm is None:
            # If no perm, reverse all dimensions (ONNX default behavior)
            input_shape = input_tensors[list(input_tensors.keys())[0]].shape
            if input_shape:
                ndim = len(input_shape)
                perm = tuple(range(ndim - 1, -1, -1))
            else:
                logger.warning(f"Transpose node {node_proto.name} has no perm and unknown input shape")
                # Default to reversing dimensions (assume 4D)
                perm = (3, 2, 1, 0)
        
        # Convert perm to list if tuple
        if isinstance(perm, tuple):
            perm = list(perm)
        
        # Find swaps needed to transform identity permutation to target permutation
        # We use a bubble sort-like approach to find the minimal swap sequence
        current = list(range(len(perm)))
        target = list(perm)
        
        # Build swap sequence to transform identity to target permutation
        swap_sequence = []
        temp_current = current[:]
        
        for i in range(len(temp_current)):
            if temp_current[i] != target[i]:
                # Find where target[i] is in current
                j = temp_current.index(target[i])
                if i != j:
                    swap_sequence.append((i, j))
                    # Perform swap
                    temp_current[i], temp_current[j] = temp_current[j], temp_current[i]
        
        # If no swaps needed, create a single identity transpose (no-op)
        if not swap_sequence:
            node_name = node_proto.name if node_proto.name else f"Transpose_{node_index}"
            # Create identity transpose (swapping first two dims, which is a no-op if they're the same)
            transpose_node = TransposeNode.create(
                name=node_name,
                inputs=list(node_proto.input),
                outputs=[node_proto.output[0]],
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                dim0=0,
                dim1=1 if len(perm) > 1 else 0
            )
            nodes.append(transpose_node)
        else:
            # Create a transpose node for each swap
            current_inputs = list(node_proto.input)
            current_input_tensors = input_tensors.copy()
            
            for swap_idx, (dim0, dim1) in enumerate(swap_sequence):
                if swap_idx == len(swap_sequence) - 1:
                    # Last swap uses the original output
                    node_outputs = [node_proto.output[0]]
                    node_output_tensors = output_tensors.copy()
                else:
                    # Intermediate swap creates a new intermediate output
                    intermediate_name = (f"{node_proto.name}_intermediate_{swap_idx}" 
                                       if node_proto.name 
                                       else f"Transpose_{node_index}_intermediate_{swap_idx}")
                    node_outputs = [intermediate_name]
                    # Create intermediate output tensor info (same shape as input, but permuted)
                    # For simplicity, use the same tensor info as input
                    input_info = list(current_input_tensors.values())[0]
                    node_output_tensors = {intermediate_name: input_info}
                
                node_name = (f"{node_proto.name}_swap_{swap_idx}" 
                           if node_proto.name 
                           else f"Transpose_{node_index}_swap_{swap_idx}")
                
                transpose_node = TransposeNode.create(
                    name=node_name,
                    inputs=current_inputs,
                    outputs=node_outputs,
                    input_tensors=current_input_tensors,
                    output_tensors=node_output_tensors,
                    dim0=dim0,
                    dim1=dim1
                )
                nodes.append(transpose_node)
                
                # Update for next iteration
                if swap_idx < len(swap_sequence) - 1:
                    current_inputs = node_outputs
                    current_input_tensors = node_output_tensors
        
        return nodes


class CastConverter(OnnxOpConverter):
    """Converter for ONNX Cast operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Cast opset v1-v8: to as attribute."""
        node_name = node_proto.name if node_proto.name else f"Cast_{node_index}"
        to_dtype = attrs.get('to', None)
        torch_dtype = None
        if to_dtype:
            # Convert ONNX dtype string to torch dtype
            if isinstance(to_dtype, str):
                # Map string dtype to ONNX enum
                dtype_map = {
                    'float32': onnx.TensorProto.FLOAT,
                    'float64': onnx.TensorProto.DOUBLE,
                    'int32': onnx.TensorProto.INT32,
                    'int64': onnx.TensorProto.INT64,
                    'bool': onnx.TensorProto.BOOL,
                }
                onnx_dtype = dtype_map.get(to_dtype, onnx.TensorProto.FLOAT)
            else:
                onnx_dtype = to_dtype
            torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)
        
        return [CastNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dtype=torch_dtype
        )]
    
    @classmethod
    def _impl_v9(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """Cast opset v9+: to as attribute (same as v1)."""
        return cls._impl_v1(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _impl_v13(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """Cast opset v13+: to as optional input tensor."""
        node_name = node_proto.name if node_proto.name else f"Cast_{node_index}"
        
        # Validate and extract dtype from constant input (second input, optional) or attribute
        is_valid, to_dtype, error_msg = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
        
        # Convert to int if it's a numpy scalar
        if to_dtype is not None:
            try:
                to_dtype = int(to_dtype)
            except (ValueError, TypeError):
                pass
        
        # Fallback to attribute
        if to_dtype is None:
            to_dtype = attrs.get('to', None)
        
        torch_dtype = None
        if to_dtype:
            if isinstance(to_dtype, str):
                dtype_map = {
                    'float32': onnx.TensorProto.FLOAT,
                    'float64': onnx.TensorProto.DOUBLE,
                    'int32': onnx.TensorProto.INT32,
                    'int64': onnx.TensorProto.INT64,
                    'bool': onnx.TensorProto.BOOL,
                }
                onnx_dtype = dtype_map.get(to_dtype, onnx.TensorProto.FLOAT)
            else:
                onnx_dtype = to_dtype
            torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)
        
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]} if data_input in input_tensors else input_tensors
        
        return [CastNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input, dtype is embedded
            outputs=[node_proto.output[0]],
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            dtype=torch_dtype
        )]

