"""
ONNX Unsqueeze operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
from ....ir.types import TensorInfo
from ....ir.operations.shape import UnsqueezeNode
from .base import OnnxOpConverter
from .validation import validate_constant_input, validate_attributes, handle_validation_error


class UnsqueezeConverter(OnnxOpConverter):
    """Converter for ONNX Unsqueeze operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Unsqueeze opset v1-v12: axes as attribute.
        """
        node_name = node_proto.name if node_proto.name else f"Unsqueeze_{node_index}"
        
        # Validate required attribute
        is_valid, error_msg = validate_attributes(node_proto, attrs, required_attrs=['axes'])
        if not is_valid:
            handle_validation_error(node_proto, error_msg or "axes attribute required", strict=True)
            return []
        
        # Extract axes from attribute
        axes = attrs.get('axes', None)
        
        # Convert to dim (Forge supports single dim, but PyTorch supports tuple)
        if isinstance(axes, (list, tuple)):
            dim = tuple(axes) if len(axes) > 1 else axes[0]
        else:
            dim = axes
        
        return [UnsqueezeNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=dim
        )]
    
    @classmethod
    def _impl_v13(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Unsqueeze opset v13+: axes as input tensor (second input).
        """
        node_name = node_proto.name if node_proto.name else f"Unsqueeze_{node_index}"
        
        # Validate and extract axes from constant input (second input) or attribute
        is_valid, axes, error_msg = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
        
        # Fallback to attribute if axes not found in input
        if not is_valid or axes is None:
            axes = attrs.get('axes', None)
        
        if axes is None:
            error_msg = (
                f"Unsqueeze {node_name} (opset >= 13) requires axes as constant initializer or attribute. "
                f"Dynamic axes tensors are not yet supported."
            )
            handle_validation_error(node_proto, error_msg, strict=True)
            return []
        
        # Convert to dim
        if isinstance(axes, (list, tuple)):
            dim = tuple(axes) if len(axes) > 1 else axes[0]
        else:
            dim = axes
        
        # Create TIR node with only data input (axes is embedded)
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]} if data_input in input_tensors else input_tensors
        
        return [UnsqueezeNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input, axes is embedded
            outputs=[node_proto.output[0]],
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            dim=dim
        )]

