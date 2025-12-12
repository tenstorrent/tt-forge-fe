"""
ONNX Pooling operation converters (MaxPool, AveragePool, GlobalAveragePool).
"""
from typing import List, Dict, Any
from onnx import NodeProto
import logging
from ....ir.types import TensorInfo
from ....ir.operations.pooling import (
    MaxPool1dNode, MaxPool2dNode, MaxPool3dNode,
    AveragePool1dNode, AveragePool2dNode, AveragePool3dNode,
    GlobalAveragePoolNode
)
from ....ir.operations.other import PadNode
from .base import OnnxOpConverter
from .utils import compute_autopad_padding

logger = logging.getLogger("ForgeTranspiler")


class MaxPoolConverter(OnnxOpConverter):
    """Converter for ONNX MaxPool operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """MaxPool opset v1+: Handles AUTO_PAD and determines 1D/2D/3D."""
        nodes = []
        node_name = node_proto.name if node_proto.name else f"MaxPool_{node_index}"
        
        # Get kernel shape to determine pool dimension
        kernel_shape = attrs.get('kernel_shape', None)
        if kernel_shape is None:
            # Try to infer from input shape
            input_shape = input_tensors[list(input_tensors.keys())[0]].shape
            if input_shape and len(input_shape) >= 3:
                # Assume 2D pooling by default
                kernel_shape = (2, 2)
                logger.warning(f"Could not infer kernel_shape for MaxPool {node_name}, defaulting to 2D")
            else:
                kernel_shape = (2, 2)  # Default to 2D
        
        # Determine pool dimension and kernel_size
        if isinstance(kernel_shape, int):
            kernel_dims = 1
            kernel_size = kernel_shape
        elif isinstance(kernel_shape, (list, tuple)):
            kernel_dims = len(kernel_shape)
            if kernel_dims == 1:
                kernel_size = kernel_shape[0] if isinstance(kernel_shape, list) else kernel_shape[0]
            else:
                kernel_size = kernel_shape  # Keep as tuple for 2D/3D
        else:
            kernel_dims = 2
            kernel_size = (2, 2)  # Default
        
        # Handle AUTO_PAD
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        pool_inputs = list(node_proto.input)
        pool_input_tensors = input_tensors.copy()
        
        if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
            # Create PadNode for auto_pad
            pad_name = f"{node_name}_pad"
            pad_output = f"{node_name}_padded"
            
            input_shape = input_tensors[pool_inputs[0]].shape
            if input_shape is None:
                logger.warning(f"Cannot compute auto_pad for MaxPool {node_name} with unknown input shape")
                auto_pad = 'NOTSET'
            else:
                stride = attrs.get('strides', None)
                if stride is None:
                    stride = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
                if isinstance(stride, int):
                    stride = (stride,) * kernel_dims
                if isinstance(kernel_shape, int):
                    kernel_shape = (kernel_shape,)
                
                pads = []
                spatial_dims = input_shape[2:]  # Skip batch and channel
                for i, (in_size, k_size, s) in enumerate(zip(spatial_dims, kernel_shape, stride)):
                    pad_before, pad_after = compute_autopad_padding(
                        in_size, k_size, s, 1, auto_pad
                    )
                    pads.extend([pad_before, pad_after])
                
                # PyTorch pad format: reverse order
                pad_list = []
                for i in range(len(pads) - 2, -1, -2):
                    pad_list.extend([pads[i], pads[i + 1]])
                
                pad_output_tensors = {pad_output: input_tensors[pool_inputs[0]]}
                
                pad_node = PadNode.create(
                    name=pad_name,
                    inputs=[pool_inputs[0]],
                    outputs=[pad_output],
                    input_tensors={pool_inputs[0]: input_tensors[pool_inputs[0]]},
                    output_tensors=pad_output_tensors,
                    pad=tuple(pad_list),
                    mode='constant',
                    value=0.0
                )
                nodes.append(pad_node)
                
                pool_inputs = [pad_output]
                pool_input_tensors = {pad_output: pad_output_tensors[pad_output]}
        
        # Extract PyTorch-compatible attributes
        stride = attrs.get('stride', None)
        if stride is None:
            stride = kernel_size if isinstance(kernel_size, int) else kernel_size[0] if isinstance(kernel_size, (list, tuple)) and len(kernel_size) > 0 else kernel_size
        padding = attrs.get('pads', 0)
        if isinstance(padding, list) and len(padding) > 0:
            # Convert ONNX pads format to PyTorch format
            if len(padding) == 4:  # 2D pool
                padding = (padding[1], padding[3], padding[0], padding[2])
            elif len(padding) == 2:  # 1D pool
                padding = (padding[0], padding[1])
            elif len(padding) == 6:  # 3D pool
                padding = (padding[1], padding[4], padding[0], padding[3], padding[2], padding[5])
            else:
                padding = tuple(padding)
        elif auto_pad != 'NOTSET':
            padding = 0  # Padding already handled by PadNode
        
        dilation = attrs.get('dilations', 1)
        ceil_mode = attrs.get('ceil_mode', False)
        
        # Create appropriate MaxPool node based on dimension
        if kernel_dims == 1:
            # Ensure 1D parameters are scalars
            k_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            s = stride if isinstance(stride, int) else (stride[0] if isinstance(stride, (list, tuple)) and len(stride) > 0 else stride)
            d = dilation if isinstance(dilation, int) else (dilation[0] if isinstance(dilation, (list, tuple)) and len(dilation) > 0 else dilation)
            pool_node = MaxPool1dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=k_size,
                stride=s,
                padding=padding,
                dilation=d,
                ceil_mode=ceil_mode
            )
        elif kernel_dims == 2:
            pool_node = MaxPool2dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode
            )
        elif kernel_dims == 3:
            pool_node = MaxPool3dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode
            )
        else:
            raise ValueError(f"Unsupported MaxPool dimension: {kernel_dims}")
        
        nodes.append(pool_node)
        return nodes


class AveragePoolConverter(OnnxOpConverter):
    """Converter for ONNX AveragePool operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """AveragePool opset v1+: Handles AUTO_PAD and determines 1D/2D/3D."""
        nodes = []
        node_name = node_proto.name if node_proto.name else f"AveragePool_{node_index}"
        
        # Get kernel shape to determine pool dimension
        kernel_shape = attrs.get('kernel_shape', None)
        if kernel_shape is None:
            # Try to infer from input shape
            input_shape = input_tensors[list(input_tensors.keys())[0]].shape
            if input_shape and len(input_shape) >= 3:
                # Assume 2D pooling by default
                kernel_shape = (2, 2)
                logger.warning(f"Could not infer kernel_shape for AveragePool {node_name}, defaulting to 2D")
            else:
                kernel_shape = (2, 2)  # Default to 2D
        
        # Determine pool dimension and kernel_size
        if isinstance(kernel_shape, int):
            kernel_dims = 1
            kernel_size = kernel_shape
        elif isinstance(kernel_shape, (list, tuple)):
            kernel_dims = len(kernel_shape)
            if kernel_dims == 1:
                kernel_size = kernel_shape[0] if isinstance(kernel_shape, list) else kernel_shape[0]
            else:
                kernel_size = kernel_shape  # Keep as tuple for 2D/3D
        else:
            kernel_dims = 2
            kernel_size = (2, 2)  # Default
        
        # Handle AUTO_PAD
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        pool_inputs = list(node_proto.input)
        pool_input_tensors = input_tensors.copy()
        
        if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
            # Create PadNode for auto_pad
            pad_name = f"{node_name}_pad"
            pad_output = f"{node_name}_padded"
            
            input_shape = input_tensors[pool_inputs[0]].shape
            if input_shape is None:
                logger.warning(f"Cannot compute auto_pad for AveragePool {node_name} with unknown input shape")
                auto_pad = 'NOTSET'
            else:
                stride = attrs.get('strides', None)
                if stride is None:
                    stride = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
                if isinstance(stride, int):
                    stride = (stride,) * kernel_dims
                if isinstance(kernel_shape, int):
                    kernel_shape = (kernel_shape,)
                
                pads = []
                spatial_dims = input_shape[2:]  # Skip batch and channel
                for i, (in_size, k_size, s) in enumerate(zip(spatial_dims, kernel_shape, stride)):
                    pad_before, pad_after = compute_autopad_padding(
                        in_size, k_size, s, 1, auto_pad
                    )
                    pads.extend([pad_before, pad_after])
                
                # PyTorch pad format: reverse order
                pad_list = []
                for i in range(len(pads) - 2, -1, -2):
                    pad_list.extend([pads[i], pads[i + 1]])
                
                pad_output_tensors = {pad_output: input_tensors[pool_inputs[0]]}
                
                pad_node = PadNode.create(
                    name=pad_name,
                    inputs=[pool_inputs[0]],
                    outputs=[pad_output],
                    input_tensors={pool_inputs[0]: input_tensors[pool_inputs[0]]},
                    output_tensors=pad_output_tensors,
                    pad=tuple(pad_list),
                    mode='constant',
                    value=0.0
                )
                nodes.append(pad_node)
                
                pool_inputs = [pad_output]
                pool_input_tensors = {pad_output: pad_output_tensors[pad_output]}
        
        # Extract PyTorch-compatible attributes
        stride = attrs.get('stride', None)
        if stride is None:
            stride = kernel_size if isinstance(kernel_size, int) else kernel_size[0] if isinstance(kernel_size, (list, tuple)) and len(kernel_size) > 0 else kernel_size
        padding = attrs.get('pads', 0)
        if isinstance(padding, list) and len(padding) > 0:
            # Convert ONNX pads format to PyTorch format
            if len(padding) == 4:  # 2D pool
                padding = (padding[1], padding[3], padding[0], padding[2])
            elif len(padding) == 2:  # 1D pool
                padding = (padding[0], padding[1])
            elif len(padding) == 6:  # 3D pool
                padding = (padding[1], padding[4], padding[0], padding[3], padding[2], padding[5])
            else:
                padding = tuple(padding)
        elif auto_pad != 'NOTSET':
            padding = 0  # Padding already handled by PadNode
        
        ceil_mode = attrs.get('ceil_mode', False)
        count_include_pad = attrs.get('count_include_pad', True)
        
        # Create appropriate AveragePool node based on dimension
        if kernel_dims == 1:
            # Ensure 1D parameters are scalars
            k_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            s = stride if isinstance(stride, int) else (stride[0] if isinstance(stride, (list, tuple)) and len(stride) > 0 else stride)
            pool_node = AveragePool1dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=k_size,
                stride=s,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad
            )
        elif kernel_dims == 2:
            pool_node = AveragePool2dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad
            )
        elif kernel_dims == 3:
            pool_node = AveragePool3dNode.create(
                name=node_name,
                inputs=pool_inputs,
                outputs=[node_proto.output[0]],
                input_tensors=pool_input_tensors,
                output_tensors=output_tensors,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad
            )
        else:
            raise ValueError(f"Unsupported AveragePool dimension: {kernel_dims}")
        
        nodes.append(pool_node)
        return nodes


class GlobalAveragePoolConverter(OnnxOpConverter):
    """Converter for ONNX GlobalAveragePool operation."""
    
    @classmethod
    def _impl_v1(cls, node_proto: NodeProto, input_tensors: Dict[str, TensorInfo],
                 output_tensors: Dict[str, TensorInfo], attrs: Dict[str, Any],
                 node_index: int, graph_proto=None) -> List:
        """GlobalAveragePool opset v1+: No version differences."""
        node_name = node_proto.name if node_proto.name else f"GlobalAveragePool_{node_index}"
        return [GlobalAveragePoolNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=[node_proto.output[0]],
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]

