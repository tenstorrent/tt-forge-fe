"""
ONNX Clip operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
from ....ir.types import TensorInfo
from ....ir.operations.other import ClipNode
from .base import OnnxOpConverter
from .validation import validate_constant_input


class ClipConverter(OnnxOpConverter):
    """Converter for ONNX Clip operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Clip opset v1: min and max as attributes.
        """
        node_name = node_proto.name if node_proto.name else f"Clip_{node_index}"
        min_val = attrs.get('min', None)
        max_val = attrs.get('max', None)
        
        return [ClipNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            min_val=min_val,
            max_val=max_val
        )]
    
    @classmethod
    def _impl_v6(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Clip opset v6+: min and max as optional input tensors (second and third inputs).
        """
        node_name = node_proto.name if node_proto.name else f"Clip_{node_index}"
        
        # Validate and extract min/max from constant inputs (optional) or attributes
        is_valid_min, min_val, error_msg_min = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
        is_valid_max, max_val, error_msg_max = validate_constant_input(
            node_proto, input_index=2, graph_proto=graph_proto
        )
        
        # Convert to float if found
        if min_val is not None:
            min_val = float(min_val)
        if max_val is not None:
            max_val = float(max_val)
        
        # Fallback to attributes if not found in inputs
        if min_val is None:
            min_val = attrs.get('min', None)
        if max_val is None:
            max_val = attrs.get('max', None)
        
        # Create TIR node with only data input (min/max are embedded)
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]} if data_input in input_tensors else input_tensors
        
        return [ClipNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input, min/max are embedded
            outputs=[node_proto.output[0]],
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            min_val=min_val,
            max_val=max_val
        )]

