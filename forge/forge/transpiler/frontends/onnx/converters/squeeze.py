"""
ONNX Squeeze operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
from ....ir.types import TensorInfo
from ....ir.operations.shape import SqueezeNode
from .base import OnnxOpConverter
from .validation import validate_constant_input


class SqueezeConverter(OnnxOpConverter):
    """Converter for ONNX Squeeze operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Squeeze opset v1-v12: axes as attribute.
        """
        node_name = node_proto.name if node_proto.name else f"Squeeze_{node_index}"
        
        # Extract axes/dim from ONNX attribute
        axes = attrs.get('axes', None)
        
        # Convert to dim (Forge only supports single dim, but PyTorch supports tuple)
        if axes is not None:
            if isinstance(axes, (list, tuple)):
                if len(axes) == 1:
                    dim = axes[0]
                else:
                    # Forge limitation: only single dim supported
                    # Use first dim and log warning
                    import logging
                    logger = logging.getLogger("ForgeTranspiler")
                    logger.warning(
                        f"Squeeze {node_name} has multiple axes {axes}, "
                        f"but Forge only supports single dim. Using first axis: {axes[0]}"
                    )
                    dim = axes[0]
            else:
                dim = axes
        else:
            # No axes specified - squeeze all dims of size 1
            # Forge requires explicit dim, so infer from input shape
            input_info = list(input_tensors.values())[0] if input_tensors else None
            if input_info and input_info.shape:
                # Find first dim of size 1
                for i, s in enumerate(input_info.shape):
                    if s == 1:
                        dim = i
                        break
                else:
                    dim = 0  # Default if no size-1 dim found
            else:
                dim = 0  # Default
        
        return [SqueezeNode.create(
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
        Squeeze opset v13+: axes as input tensor (second input).
        """
        node_name = node_proto.name if node_proto.name else f"Squeeze_{node_index}"
        
        # Validate and extract axes from constant input (second input) or attribute
        is_valid, axes, error_msg = validate_constant_input(
            node_proto, input_index=1, graph_proto=graph_proto
        )
        
        # Convert to tuple if it's a list
        if axes is not None and isinstance(axes, list):
            axes = tuple(int(x) for x in axes)
        elif axes is not None:
            axes = tuple([int(axes)]) if not isinstance(axes, (list, tuple)) else tuple(int(x) for x in axes)
        
        # Fallback to attribute if axes not found in input
        if axes is None:
            axes = attrs.get('axes', None)
        
        # Convert to dim (Forge only supports single dim)
        if axes is not None:
            if isinstance(axes, (list, tuple)):
                if len(axes) == 1:
                    dim = axes[0]
                else:
                    # Forge limitation: only single dim supported
                    import logging
                    logger = logging.getLogger("ForgeTranspiler")
                    logger.warning(
                        f"Squeeze {node_name} has multiple axes {axes}, "
                        f"but Forge only supports single dim. Using first axis: {axes[0]}"
                    )
                    dim = axes[0]
            else:
                dim = axes
        else:
            # No axes specified - squeeze all dims of size 1
            # Forge requires explicit dim, so infer from input shape
            data_input = node_proto.input[0]
            input_info = input_tensors.get(data_input) if input_tensors else None
            if input_info and input_info.shape:
                # Find first dim of size 1
                for i, s in enumerate(input_info.shape):
                    if s == 1:
                        dim = i
                        break
                else:
                    dim = 0  # Default if no size-1 dim found
            else:
                dim = 0  # Default
        
        # Create TIR node with only data input (axes is embedded)
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]} if data_input in input_tensors else input_tensors
        
        return [SqueezeNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input, axes is embedded
            outputs=[node_proto.output[0]],
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            dim=dim
        )]

