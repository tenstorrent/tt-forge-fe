"""
ONNX Shape operation converters (Transpose, Cast).
"""
from typing import List, Dict, Any, Tuple
from onnx import NodeProto
import onnx
from ....ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from ....ir.operations.shape import TransposeNode
from ....ir.operations.other import CastNode
from .base import OnnxOpConverter
from .validation import validate_constant_input


class TransposeConverter(OnnxOpConverter):
    """Converter for ONNX Transpose operation."""
    
    @classmethod
    def _normalize_perm(cls, perm: Any, input_rank: int) -> List[int]:
        """Normalize permutation to positive integers, handling negative indices."""
        if perm is None:
            return list(range(input_rank - 1, -1, -1))
        return [idx + input_rank if idx < 0 else idx for idx in map(int, perm)]
    
    @classmethod
    def _validate_perm(cls, perm: List[int], input_rank: int) -> None:
        """Validate that perm is a valid permutation of [0, 1, ..., n-1]."""
        if len(perm) != input_rank:
            raise ValueError(f"Permutation length ({len(perm)}) must equal input rank ({input_rank}). Perm: {perm}")
        
        if set(perm) != set(range(input_rank)):
            raise ValueError(f"Permutation must be a permutation of [0, 1, ..., {input_rank-1}]. Got: {perm}")
    
    @classmethod
    def _is_identity_perm(cls, perm: List[int]) -> bool:
        """Check if permutation is identity (no change)."""
        return all(i == idx for i, idx in enumerate(perm))
    
    @classmethod
    def _decompose_permutation(cls, perm: List[int]) -> List[Tuple[int, int]]:
        """
        Decompose a permutation into a sequence of two-dimension swaps.
        
        Uses a greedy algorithm that tracks the current state after each swap.
        For each position i, if the current value at i is not the target value,
        find where the target value is and swap it into place.
        
        Args:
            perm: Target permutation (list of integers)
            
        Returns:
            List of (dim0, dim1) tuples representing swap operations in the
            current coordinate system (each swap is applied to the result of previous swaps)
        """
        swap_sequence = []
        # Work with a mutable copy representing current state
        # current[i] = which original dimension is currently at position i
        current = list(range(len(perm)))
        target = list(perm)
        
        # For each position, bring the correct value into place
        for i in range(len(target)):
            # What value should be at position i in the final result?
            target_value = target[i]
            
            # What value is currently at position i?
            current_value = current[i]
            
            if current_value != target_value:
                # Find where target_value currently is
                j = current.index(target_value)
                if i != j:
                    # Swap positions i and j
                    swap_sequence.append((i, j))
                    # Update current state to reflect the swap
                    current[i], current[j] = current[j], current[i]
        
        return swap_sequence
    
    @classmethod
    def _convert_permutation_to_swaps(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                                      output_tensors: Dict[str, TensorInfo], perm: List[int],
                                      node_index: int) -> List:
        """Convert ONNX permutation to a sequence of TransposeNode operations."""
        from ....ir.operations.other import IdentityNode
        
        # Check if identity permutation or empty swap sequence
        swap_sequence = cls._decompose_permutation(perm) if not cls._is_identity_perm(perm) else []
        if not swap_sequence:
            node_name = node_proto.name or f"Transpose_{node_index}"
            return [IdentityNode.create(
                name=node_name,
                inputs=list(node_proto.input),
                outputs=[node_proto.output[0]],
                input_tensors=input_tensors,
                output_tensors=output_tensors
            )]
        
        # Create transpose nodes for each swap
        nodes = []
        input_info = list(input_tensors.values())[0]
        current_inputs = list(node_proto.input)
        current_input_tensors = input_tensors.copy()
        current_shape = list(input_info.shape) if input_info.shape else None
        base_name = node_proto.name or f"Transpose_{node_index}"
        onnx_dtype = getattr(input_info, 'onnx_dtype', None)
        
        for swap_idx, (dim0, dim1) in enumerate(swap_sequence):
            is_last = swap_idx == len(swap_sequence) - 1
            
            if is_last:
                node_outputs = [node_proto.output[0]]
                node_output_tensors = output_tensors.copy()
            else:
                intermediate_name = f"{base_name}_intermediate_{swap_idx}"
                node_outputs = [intermediate_name]
                if current_shape is not None:
                    new_shape = list(current_shape)
                    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
                    intermediate_shape = tuple(new_shape)
                else:
                    intermediate_shape = None
                node_output_tensors = {
                    intermediate_name: TensorInfo(name=intermediate_name, shape=intermediate_shape, onnx_dtype=onnx_dtype)
                }
            
            nodes.append(TransposeNode.create(
                name=f"{base_name}_swap_{swap_idx}",
                inputs=current_inputs,
                outputs=node_outputs,
                input_tensors=current_input_tensors,
                output_tensors=node_output_tensors,
                dim0=dim0,
                dim1=dim1
            ))
            
            if not is_last:
                current_inputs = node_outputs
                current_input_tensors = node_output_tensors
                if current_shape is not None:
                    current_shape[dim0], current_shape[dim1] = current_shape[dim1], current_shape[dim0]
        
        return nodes
    
    @classmethod
    def _get_input_rank(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                        output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any]) -> int:
        """Determine input rank from available information."""
        input_info = list(input_tensors.values())[0]
        if input_info.shape:
            return len(input_info.shape)
        
        output_info = list(output_tensors.values())[0]
        if output_info.shape:
            return len(output_info.shape)
        
        perm = attrs.get('perm')
        if perm:
            return len(perm)
        
        raise ValueError(
            f"Cannot determine input rank for Transpose node '{node_proto.name}'. "
            f"Input shape, output shape, and perm attribute are all unknown."
        )
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Transpose opset v1+: No version differences.
        
        ONNX Transpose uses perm=[0,2,1,3] which can swap multiple dimensions.
        PyTorch transpose only swaps two dimensions, so we decompose the permutation
        into a series of two-dimension swaps.
        
        If perm is omitted, defaults to (n-1, ..., 0) (reverse all dimensions).
        """
        input_rank = cls._get_input_rank(node_proto, input_tensors, output_tensors, attrs)
        perm = cls._normalize_perm(attrs.get('perm', None), input_rank)
        cls._validate_perm(perm, input_rank)
        return cls._convert_permutation_to_swaps(
            node_proto, input_tensors, output_tensors, perm, node_index
        )
    
    # All versions use the same implementation (no functional differences)
    _impl_v13 = _impl_v21 = _impl_v23 = _impl_v24 = _impl_v25 = _impl_v1


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

