"""
ONNX Conv operation converter with opset version support.
"""
from typing import List, Dict, Any
from onnx import NodeProto
from ....ir.types import TensorInfo
from ....ir.operations.conv import Conv1dNode, Conv2dNode, Conv3dNode
from ....ir.operations.other import PadNode
from .base import OnnxOpConverter
from .autopad import AutoPad
import logging

logger = logging.getLogger("ForgeTranspiler")


class ConvConverter(OnnxOpConverter):
    """Converter for ONNX Conv operation with opset version support."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """
        Conv opset v1-v10: groups as attribute (default 1).
        """
        return cls._convert_conv_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _impl_v11(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                  output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                  node_index: int, graph_proto=None) -> List:
        """
        Conv opset v11+: groups as attribute (same as v1, but with better validation).
        """
        return cls._convert_conv_impl(node_proto, input_tensors, output_tensors, attrs, node_index, graph_proto)
    
    @classmethod
    def _convert_conv_impl(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                          output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                          node_index: int, graph_proto=None) -> List:
        """
        Common implementation for Conv conversion.
        """
        nodes = []
        node_name = node_proto.name if node_proto.name else f"Conv_{node_index}"
        
        # Get kernel shape to determine conv dimension
        kernel_shape = attrs.get('kernel_shape', None)
        if kernel_shape is None:
            # Try to infer from weight tensor shape
            weight_name = node_proto.input[1] if len(node_proto.input) > 1 else None
            if weight_name and weight_name in input_tensors:
                weight_shape = input_tensors[weight_name].shape
                if weight_shape and len(weight_shape) >= 2:
                    # ONNX Conv weight shape: [out_channels, in_channels, *kernel_dims]
                    kernel_shape = weight_shape[2:]
                else:
                    logger.warning(f"Could not infer kernel_shape for Conv {node_name}, defaulting to Conv2d")
                    kernel_shape = (3, 3)  # Default to 2D
            else:
                logger.warning(f"Could not infer kernel_shape for Conv {node_name}, defaulting to Conv2d")
                kernel_shape = (3, 3)  # Default to 2D
        
        # Determine conv dimension
        if isinstance(kernel_shape, int):
            kernel_dims = 1
            kernel_size = kernel_shape
        else:
            kernel_dims = len(kernel_shape)
            kernel_size = kernel_shape[0] if len(kernel_shape) == 1 else kernel_shape
        
        # Handle AUTO_PAD
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        pad_node = None
        conv_inputs = list(node_proto.input)
        conv_input_tensors = input_tensors.copy()
        
        if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
            # Create PadNode for auto_pad
            pad_name = f"{node_name}_pad"
            pad_output = f"{node_name}_padded"
            
            # Get input shape to compute padding
            input_shape = input_tensors[conv_inputs[0]].shape
            if input_shape is None:
                logger.warning(f"Cannot compute auto_pad for Conv {node_name} with unknown input shape")
                auto_pad = 'NOTSET'
            else:
                # Compute padding values
                stride = attrs.get('strides', 1)
                dilation = attrs.get('dilations', 1)
                
                if isinstance(stride, int):
                    stride = (stride,) * kernel_dims
                if isinstance(dilation, int):
                    dilation = (dilation,) * kernel_dims
                if isinstance(kernel_shape, int):
                    kernel_shape = (kernel_shape,)
                
                # Compute padding for each spatial dimension
                pads = []
                spatial_dims = input_shape[2:]  # Skip batch and channel
                for i, (in_size, k_size, s, d) in enumerate(zip(spatial_dims, kernel_shape, stride, dilation)):
                    pad_before, pad_after = AutoPad.compute_padding(
                        in_size, k_size, s, d, auto_pad
                    )
                    pads.extend([pad_before, pad_after])
                
                # PyTorch pad format: [left, right, top, bottom] for 2D, etc.
                # Reverse order for F.pad
                pad_list = []
                for i in range(len(pads) - 2, -1, -2):
                    pad_list.extend([pads[i], pads[i + 1]])
                
                # Create pad output tensor info
                pad_output_tensors = {pad_output: input_tensors[conv_inputs[0]]}
                
                pad_node = PadNode.create(
                    name=pad_name,
                    inputs=[conv_inputs[0]],
                    outputs=[pad_output],
                    input_tensors={conv_inputs[0]: input_tensors[conv_inputs[0]]},
                    output_tensors=pad_output_tensors,
                    pad=tuple(pad_list),
                    mode='constant',
                    value=0.0
                )
                nodes.append(pad_node)
                
                # Conv will use padded output
                conv_inputs = [pad_output]
                conv_input_tensors = {pad_output: pad_output_tensors[pad_output]}
        
        # Extract PyTorch-compatible attributes
        stride = attrs.get('strides', 1)
        padding = attrs.get('pads', 0)
        if isinstance(padding, list) and len(padding) > 0:
            # ONNX pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            # PyTorch format: (left, right, top, bottom) for 2D
            if len(padding) == 4:  # 2D conv
                padding = (padding[1], padding[3], padding[0], padding[2])  # Reorder to PyTorch format
            elif len(padding) == 2:  # 1D conv
                padding = (padding[0], padding[1])
            elif len(padding) == 6:  # 3D conv
                padding = (padding[1], padding[4], padding[0], padding[3], padding[2], padding[5])
            else:
                padding = tuple(padding)
        elif auto_pad != 'NOTSET':
            padding = 0  # Padding already handled by PadNode
        
        dilation = attrs.get('dilations', 1)
        groups = attrs.get('group', 1)
        
        # Create appropriate Conv node
        conv_output_tensors = output_tensors.copy()
        if kernel_dims == 1:
            conv_node = Conv1dNode.create(
                name=node_name,
                inputs=conv_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=conv_input_tensors,
                output_tensors=conv_output_tensors,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups
            )
        elif kernel_dims == 2:
            conv_node = Conv2dNode.create(
                name=node_name,
                inputs=conv_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=conv_input_tensors,
                output_tensors=conv_output_tensors,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups
            )
        elif kernel_dims == 3:
            conv_node = Conv3dNode.create(
                name=node_name,
                inputs=conv_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=conv_input_tensors,
                output_tensors=conv_output_tensors,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups
            )
        else:
            raise ValueError(f"Unsupported Conv dimension: {kernel_dims}")
        
        nodes.append(conv_node)
        return nodes

