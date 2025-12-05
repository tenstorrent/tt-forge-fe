"""
ONNX Reshape operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
from ....ir.types import TensorInfo
from ....ir.operations.shape import ReshapeNode
from .base import OnnxOpConverter
from .validation import validate_constant_input, handle_validation_error


class ReshapeConverter(OnnxOpConverter):
    """Converter for ONNX Reshape operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Reshape opset v1-v4: shape as attribute.
        """
        node_name = node_proto.name if node_proto.name else f"Reshape_{node_index}"
        
        # Extract shape from attribute
        shape = attrs.get('shape', None)
        if shape is None:
            error_msg = f"Reshape {node_name} (opset < 5) requires 'shape' attribute"
            handle_validation_error(node_proto, error_msg, strict=True)
            return []
        
        return [ReshapeNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            shape=tuple(shape) if isinstance(shape, list) else shape
        )]
    
    @classmethod
    def _impl_v5(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Reshape opset v5+: shape as input tensor (second input).
        """
        node_name = node_proto.name if node_proto.name else f"Reshape_{node_index}"
        
        # ONNX Reshape has 2 inputs: [data, shape]
        # TIR Reshape has 1 input: [data], shape is an attribute
        data_input = node_proto.input[0]
        
        # Validate and extract shape from constant input (second input)
        is_valid, shape_value, error_msg = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
        
        if not is_valid:
            # If shape input is optional and not provided, try to get from output shape
            if shape_value is None and len(node_proto.output) > 0:
                output_info = output_tensors.get(node_proto.output[0])
                if output_info and output_info.shape:
                    shape_value = output_info.shape
                else:
                    handle_validation_error(node_proto, error_msg or "Shape input required", strict=True)
                    return []
            else:
                handle_validation_error(node_proto, error_msg or "Shape input required", strict=True)
                return []
        
        # Convert shape_value to tuple if it's a list
        if isinstance(shape_value, list):
            shape_value = tuple(int(x) for x in shape_value)
        elif shape_value is not None:
            shape_value = tuple(int(x) for x in [shape_value]) if not isinstance(shape_value, tuple) else tuple(int(x) for x in shape_value)
        
        # Create TIR node with only data input
        tir_input_tensors = {data_input: input_tensors[data_input]} if data_input in input_tensors else input_tensors
        
        return [ReshapeNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input, shape is embedded
            outputs=[node_proto.output[0]],
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            shape=shape_value
        )]

