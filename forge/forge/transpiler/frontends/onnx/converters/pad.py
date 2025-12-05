"""
ONNX Pad operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
from ....ir.types import TensorInfo
from ....ir.operations.other import PadNode
from .base import OnnxOpConverter
from .validation import validate_constant_input, handle_validation_error


class PadConverter(OnnxOpConverter):
    """Converter for ONNX Pad operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo], 
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any], 
                  node_index: int, graph_proto=None) -> List:
        """
        Pad opset v1-v2: pads as attribute.
        """
        node_name = node_proto.name if node_proto.name else f"Pad_{node_index}"
        pads = attrs.get('pads', [])
        mode = attrs.get('mode', 'constant')
        value = attrs.get('value', 0.0)
        
        # Convert ONNX pads format to PyTorch format
        if isinstance(pads, list) and len(pads) > 0:
            # ONNX: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            # PyTorch: reverse order, pairs for each dimension
            pad_list = []
            ndim = len(pads) // 2
            for i in range(ndim - 1, -1, -1):
                pad_list.extend([pads[i], pads[i + ndim]])
            pads = tuple(pad_list)
        else:
            pads = tuple(pads) if isinstance(pads, list) else pads
        
        return [PadNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            pad=pads,
            mode=mode,
            value=value
        )]
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Pad opset v11+: pads as input tensor (second input).
        If pads is not provided as input, falls back to attribute.
        """
        node_name = node_proto.name if node_proto.name else f"Pad_{node_index}"
        mode = attrs.get('mode', 'constant')
        value = attrs.get('value', 0.0)
        
        # Validate and extract pads from constant input (second input)
        is_valid, pads, error_msg = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
        
        # Convert to list if it's a tuple or numpy array
        if pads is not None:
            if isinstance(pads, (list, tuple)):
                pads = [int(x) for x in pads]
            else:
                pads = [int(pads)]
        
        # Fallback to attribute if pads not found in input
        if pads is None:
            pads = attrs.get('pads', [])
        
        # Validate that pads is provided
        if not pads:
            handle_validation_error(
                node_proto, 
                f"Pad {node_proto.name or 'Pad'} requires 'pads' attribute or input",
                strict=True
            )
            return []
        
        # Convert ONNX pads format to PyTorch format
        if isinstance(pads, list) and len(pads) > 0:
            # ONNX: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            # PyTorch: reverse order, pairs for each dimension
            pad_list = []
            ndim = len(pads) // 2
            for i in range(ndim - 1, -1, -1):
                pad_list.extend([pads[i], pads[i + ndim]])
            pads = tuple(pad_list)
        else:
            pads = tuple(pads) if isinstance(pads, list) else pads
        
        return [PadNode.create(
            name=node_name,
            inputs=[node_proto.input[0]],  # Only data input, pads is embedded
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            pad=pads,
            mode=mode,
            value=value
        )]

