"""
ONNX transpiler engine for converting ONNX models to Forge graphs.
"""
import onnx
from onnx import numpy_helper, shape_inference
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from typing import List as ListType

from ...ir.types import TensorInfo, onnx_dtype_to_torch_dtype
from ...ir.nodes import TIRNode
from ...ir.operations.generic import GenericNode
from ...ir.operations.arithmetic import AddNode, SubNode, MulNode, DivNode, MatMulNode
from ...ir.operations.conv import Conv1dNode, Conv2dNode, Conv3dNode
from ...ir.operations.shape import TransposeNode, ReshapeNode, SqueezeNode
from ...ir.operations.other import ConcatNode, ClipNode, CastNode, PadNode
from ...ir.operations.reduction import ReduceSumNode, ReduceMeanNode, ReduceMaxNode
from ...ir.operations.activation import ReluNode, SigmoidNode, TanhNode, SoftmaxNode, LogSoftmaxNode, LeakyReluNode
from ...ir.operations.pooling import MaxPoolNode, AveragePoolNode, GlobalAveragePoolNode
from ...ir.operations.normalization import BatchNormalizationNode
from ...core.graph import TIRGraph
from .converters import (
    extract_attributes,
    remove_initializers_from_input,
    get_inputs_names,
    get_outputs_names,
)
from .converters.autopad import AutoPad

logger = logging.getLogger("ForgeTranspiler")


class ONNXToForgeTranspiler:
    """Main transpiler class for converting ONNX models to Forge graphs."""
    def __init__(self, debug: bool = False):
        """
        Initialize the transpiler.
        
        Args:
            debug: Enable debug mode (compare outputs with ONNXRuntime)
        """
        self.debug = debug
        self.onnx_model = None  # Store original model for debug mode
        if debug:
            try:
                import onnxruntime
            except ImportError:
                logger.warning("onnxruntime not available. Debug mode requires: pip install onnxruntime")
                self.debug = False
        
        # Dictionary mapping ONNX op types to converter methods
        self._op_converters = {
            "Add": self._convert_add,
            "Sub": self._convert_sub,
            "Mul": self._convert_mul,
            "Div": self._convert_div,
            "MatMul": self._convert_matmul,
            "Transpose": self._convert_transpose,
            "Conv": self._convert_conv,
            "Concat": self._convert_concat,
            "Clip": self._convert_clip,
            "Cast": self._convert_cast,
            "Pad": self._convert_pad,
            "Reshape": self._convert_reshape,
            "Squeeze": self._convert_squeeze,
            "ReduceSum": self._convert_reduce_sum,
            "ReduceMean": self._convert_reduce_mean,
            "ReduceMax": self._convert_reduce_max,
            "Relu": self._convert_relu,
            "Sigmoid": self._convert_sigmoid,
            "Tanh": self._convert_tanh,
            "Softmax": self._convert_softmax,
            "LogSoftmax": self._convert_log_softmax,
            "LeakyRelu": self._convert_leaky_relu,
            "MaxPool": self._convert_max_pool,
            "AveragePool": self._convert_average_pool,
            "GlobalAveragePool": self._convert_global_average_pool,
            "BatchNormalization": self._convert_batch_normalization,
        }

    def _get_tensor_info(self, value_info_map, name):
        """
        Retrieves shape and dtype and wraps it in a TensorInfo object.
        """
        if name not in value_info_map:
            # Handle the case where the name is an input that is not defined in value_info 
            return TensorInfo(name, None, onnx.TensorProto.UNDEFINED)
        
        vi = value_info_map[name]
        tensor_type = vi.type.tensor_type
        
        onnx_dtype = tensor_type.elem_type
        shape = None

        if tensor_type.HasField("shape"):
            shape = []
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param) # Represents dynamic dimension name
                else:
                    shape.append(None) # Represents unknown dynamic dimension
            shape = tuple(shape)
        
        return TensorInfo(name, shape, onnx_dtype)
    
    def _convert_transpose(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """
        Convert ONNX Transpose to one or more TransposeNode instances.
        ONNX Transpose uses perm=[0,2,1,3] which swaps multiple dimensions.
        PyTorch transpose only swaps two dimensions, so we create multiple nodes.
        """
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
                # Default to reversing dimensions
                perm = tuple(range(len(node_proto.input) - 1, -1, -1))
        
        # Convert perm to identity permutation to find swaps needed
        # We need to decompose the permutation into a series of two-dimension swaps
        current = list(range(len(perm)))
        target = list(perm)
        
        # Find swaps needed using bubble sort-like approach
        swap_sequence = []
        temp_current = current[:]
        temp_target = target[:]
        
        # Build swap sequence to transform identity to target permutation
        for i in range(len(temp_current)):
            if temp_current[i] != temp_target[i]:
                # Find where target[i] is in current
                j = temp_current.index(temp_target[i])
                if i != j:
                    swap_sequence.append((i, j))
                    # Perform swap
                    temp_current[i], temp_current[j] = temp_current[j], temp_current[i]
        
        # If no swaps needed, create a single identity transpose
        if not swap_sequence:
            node_name = node_proto.name if node_proto.name else f"Transpose_{node_index}"
            # Create identity transpose (swapping first two dims, which is a no-op if they're the same)
            transpose_node = TransposeNode.create(
                name=node_name,
                inputs=list(node_proto.input),
                outputs=list(node_proto.output),
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
                    node_outputs = list(node_proto.output)
                    node_output_tensors = output_tensors.copy()
                else:
                    # Intermediate swap creates a new intermediate output
                    intermediate_name = f"{node_proto.name}_intermediate_{swap_idx}" if node_proto.name else f"Transpose_{node_index}_intermediate_{swap_idx}"
                    node_outputs = [intermediate_name]
                    # Create intermediate output tensor info (same as input for now)
                    node_output_tensors = {intermediate_name: current_input_tensors[list(current_input_tensors.keys())[0]]}
                
                node_name = f"{node_proto.name}_swap_{swap_idx}" if node_proto.name else f"Transpose_{node_index}_swap_{swap_idx}"
                
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
                
                # Next transpose uses this output as input
                current_inputs = node_outputs
                current_input_tensors = node_output_tensors
        
        return nodes
    
    def _convert_conv(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """
        Convert ONNX Conv to Conv1d/2d/3d based on kernel_size.
        If AUTO_PAD is set, create PadNode first, then ConvNdNode.
        """
        nodes = []
        node_name = node_proto.name if node_proto.name else f"Conv_{node_index}"
        
        # Get kernel shape to determine conv dimension
        kernel_shape = attrs.get('kernel_size', None)
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
                stride = attrs.get('stride', 1)
                dilation = attrs.get('dilation', 1)
                
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
        stride = attrs.get('stride', 1)
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
        
        dilation = attrs.get('dilation', 1)
        groups = attrs.get('groups', 1)
        
        # Create appropriate Conv node
        conv_output_tensors = output_tensors.copy()
        if kernel_dims == 1:
            conv_node = Conv1dNode.create(
                name=node_name,
                inputs=conv_inputs,
                outputs=list(node_proto.output),
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
                outputs=list(node_proto.output),
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
                outputs=list(node_proto.output),
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
    
    def _convert_add(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Add to AddNode."""
        node_name = node_proto.name if node_proto.name else f"Add_{node_index}"
        return [AddNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]
    
    def _convert_sub(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Sub to SubNode."""
        node_name = node_proto.name if node_proto.name else f"Sub_{node_index}"
        return [SubNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]
    
    def _convert_mul(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Mul to MulNode."""
        node_name = node_proto.name if node_proto.name else f"Mul_{node_index}"
        return [MulNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]
    
    def _convert_div(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Div to DivNode."""
        node_name = node_proto.name if node_proto.name else f"Div_{node_index}"
        return [DivNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]
    
    def _convert_matmul(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX MatMul to MatMulNode."""
        node_name = node_proto.name if node_proto.name else f"MatMul_{node_index}"
        return [MatMulNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]
    
    def _convert_concat(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Concat to ConcatNode."""
        node_name = node_proto.name if node_proto.name else f"Concat_{node_index}"
        dim = attrs.get('axis', 0)
        return [ConcatNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=dim
        )]
    
    def _convert_clip(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """
        Convert ONNX Clip to ClipNode.
        ONNX Clip: Can have min/max as attributes or as input tensors (2nd and 3rd inputs)
        PyTorch/TIR Clip: 1 input, min/max as attributes
        Forge Clip: 1 input, min/max as attributes
        """
        node_name = node_proto.name if node_proto.name else f"Clip_{node_index}"
        
        # Extract min/max from attributes first
        min_val = attrs.get('min', None)
        max_val = attrs.get('max', None)
        
        # If not in attrs, check if provided as input tensors (opset >= 11)
        if min_val is None and len(node_proto.input) > 1:
            min_input = node_proto.input[1]
            # Try to get min from initializers
            if hasattr(self, 'graph_proto') and self.graph_proto:
                for init in self.graph_proto.initializer:
                    if init.name == min_input:
                        from onnx import numpy_helper
                        min_array = numpy_helper.to_array(init)
                        min_val = float(min_array.item())
                        break
        
        if max_val is None and len(node_proto.input) > 2:
            max_input = node_proto.input[2]
            # Try to get max from initializers
            if hasattr(self, 'graph_proto') and self.graph_proto:
                for init in self.graph_proto.initializer:
                    if init.name == max_input:
                        from onnx import numpy_helper
                        max_array = numpy_helper.to_array(init)
                        max_val = float(max_array.item())
                        break
        
        # Create TIR node with only data input
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]}
        
        return [ClipNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input
            outputs=list(node_proto.output),
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            min_val=min_val,
            max_val=max_val
        )]
    
    def _convert_cast(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Cast to CastNode."""
        node_name = node_proto.name if node_proto.name else f"Cast_{node_index}"
        to_dtype = attrs.get('to', None)
        torch_dtype = None
        if to_dtype:
            # Convert ONNX dtype string to torch dtype
            from ...ir.types import onnx_dtype_to_torch_dtype
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
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dtype=torch_dtype
        )]
    
    def _convert_pad(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Pad to PadNode."""
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
        
        return [PadNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            pad=pads,
            mode=mode,
            value=value
        )]
    
    def _convert_reshape(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """
        Convert ONNX Reshape to ReshapeNode.
        ONNX Reshape takes 2 inputs: (data, shape) where shape is a tensor.
        PyTorch/TIR Reshape takes 1 input and shape as attribute.
        We extract shape from ONNX's second input (from initializers or previous nodes).
        """
        node_name = node_proto.name if node_proto.name else f"Reshape_{node_index}"
        
        # ONNX Reshape has 2 inputs: [data, shape]
        # TIR Reshape has 1 input: [data], shape is an attribute
        data_input = node_proto.input[0]
        shape_input = node_proto.input[1] if len(node_proto.input) > 1 else None
        
        # Extract shape value from ONNX's second input
        shape_value = None
        if shape_input:
            # Try to get shape from initializers (if it's a constant)
            if hasattr(self, 'graph_proto') and self.graph_proto:
                for init in self.graph_proto.initializer:
                    if init.name == shape_input:
                        # Shape is a constant initializer
                        from onnx import numpy_helper
                        shape_array = numpy_helper.to_array(init)
                        shape_value = tuple(int(x) for x in shape_array.tolist())
                        break
            
            # If not found in initializers, try to infer from output shape
            if shape_value is None:
                output_info = list(output_tensors.values())[0]
                if output_info and output_info.shape:
                    shape_value = tuple(output_info.shape)
        
        # If still no shape, use output shape or default
        if shape_value is None:
            output_info = list(output_tensors.values())[0]
            if output_info and output_info.shape:
                shape_value = tuple(output_info.shape)
            else:
                logger.warning(f"Could not determine shape for Reshape {node_name}, will use output shape at runtime")
                shape_value = None  # Will be handled in eval()
        
        # Create TIR node with only data input and shape as attribute
        tir_input_tensors = {data_input: input_tensors[data_input]}
        
        return [ReshapeNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input
            outputs=list(node_proto.output),
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            shape=shape_value if shape_value else tuple()  # Will be resolved at runtime if None
        )]
    
    def _convert_reduce_sum(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """
        Convert ONNX ReduceSum to ReduceSumNode.
        ONNX ReduceSum: axes as attribute (or input tensor in newer opsets)
        PyTorch/TIR ReduceSum: 1 input, dim as attribute
        """
        node_name = node_proto.name if node_proto.name else f"ReduceSum_{node_index}"
        
        # Extract axes/dim from ONNX
        axes = attrs.get('axes', None)
        keepdims = attrs.get('keepdims', False)
        
        # Convert axes to dim (PyTorch uses dim)
        dim = None
        if axes is not None:
            if isinstance(axes, (list, tuple)):
                dim = tuple(axes) if len(axes) > 1 else (axes[0] if len(axes) == 1 else None)
            else:
                dim = axes
        
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]}
        
        return [ReduceSumNode.create(
            name=node_name,
            inputs=[data_input],
            outputs=list(node_proto.output),
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            dim=dim,
            keepdim=keepdims
        )]
    
    def _convert_reduce_mean(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX ReduceMean to ReduceMeanNode."""
        node_name = node_proto.name if node_proto.name else f"ReduceMean_{node_index}"
        
        axes = attrs.get('axes', None)
        keepdims = attrs.get('keepdims', False)
        
        dim = None
        if axes is not None:
            if isinstance(axes, (list, tuple)):
                dim = tuple(axes) if len(axes) > 1 else (axes[0] if len(axes) == 1 else None)
            else:
                dim = axes
        
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]}
        
        return [ReduceMeanNode.create(
            name=node_name,
            inputs=[data_input],
            outputs=list(node_proto.output),
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            dim=dim,
            keepdim=keepdims
        )]
    
    def _convert_reduce_max(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX ReduceMax to ReduceMaxNode."""
        node_name = node_proto.name if node_proto.name else f"ReduceMax_{node_index}"
        
        axes = attrs.get('axes', None)
        keepdims = attrs.get('keepdims', False)
        
        dim = None
        if axes is not None:
            if isinstance(axes, (list, tuple)):
                dim = tuple(axes) if len(axes) > 1 else (axes[0] if len(axes) == 1 else None)
            else:
                dim = axes
        
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]}
        
        return [ReduceMaxNode.create(
            name=node_name,
            inputs=[data_input],
            outputs=list(node_proto.output),
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            dim=dim,
            keepdim=keepdims
        )]
    
    def _convert_relu(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Relu to ReluNode."""
        node_name = node_proto.name if node_proto.name else f"Relu_{node_index}"
        return [ReluNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]
    
    def _convert_sigmoid(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Sigmoid to SigmoidNode."""
        node_name = node_proto.name if node_proto.name else f"Sigmoid_{node_index}"
        return [SigmoidNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]
    
    def _convert_tanh(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Tanh to TanhNode."""
        node_name = node_proto.name if node_proto.name else f"Tanh_{node_index}"
        return [TanhNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]
    
    def _convert_softmax(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX Softmax to SoftmaxNode."""
        node_name = node_proto.name if node_proto.name else f"Softmax_{node_index}"
        # ONNX uses 'axis', PyTorch uses 'dim'
        axis = attrs.get('axis', -1)
        return [SoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]
    
    def _convert_log_softmax(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX LogSoftmax to LogSoftmaxNode."""
        node_name = node_proto.name if node_proto.name else f"LogSoftmax_{node_index}"
        axis = attrs.get('axis', -1)
        return [LogSoftmaxNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            dim=axis
        )]
    
    def _convert_leaky_relu(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX LeakyRelu to LeakyReluNode."""
        node_name = node_proto.name if node_proto.name else f"LeakyRelu_{node_index}"
        # ONNX uses 'alpha', PyTorch uses 'negative_slope'
        alpha = attrs.get('alpha', 0.01)
        return [LeakyReluNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            negative_slope=alpha
        )]
    
    def _convert_max_pool(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """
        Convert ONNX MaxPool to MaxPoolNode.
        Handles AUTO_PAD by creating PadNode first if needed.
        Maps to MaxPool1d/2d/3d based on kernel_size.
        """
        nodes = []
        node_name = node_proto.name if node_proto.name else f"MaxPool_{node_index}"
        
        # Get kernel shape to determine pool dimension
        kernel_shape = attrs.get('kernel_size', attrs.get('kernel_shape', None))
        if kernel_shape is None:
            logger.warning(f"Could not infer kernel_shape for MaxPool {node_name}, defaulting to 2D")
            kernel_shape = (2, 2)  # Default to 2D
        
        if isinstance(kernel_shape, int):
            kernel_dims = 1
            kernel_size = kernel_shape
        else:
            kernel_dims = len(kernel_shape)
            kernel_size = kernel_shape[0] if len(kernel_shape) == 1 else kernel_shape
        
        # Handle AUTO_PAD
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        pad_node = None
        pool_inputs = list(node_proto.input)
        pool_input_tensors = input_tensors.copy()
        
        if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
            # Create PadNode for auto_pad (similar to Conv)
            pad_name = f"{node_name}_pad"
            pad_output = f"{node_name}_padded"
            
            input_shape = input_tensors[pool_inputs[0]].shape
            if input_shape is None:
                logger.warning(f"Cannot compute auto_pad for MaxPool {node_name} with unknown input shape")
                auto_pad = 'NOTSET'
            else:
                stride = attrs.get('stride', kernel_size)
                if isinstance(stride, int):
                    stride = (stride,) * kernel_dims
                if isinstance(kernel_shape, int):
                    kernel_shape = (kernel_shape,)
                
                pads = []
                spatial_dims = input_shape[2:]  # Skip batch and channel
                for i, (in_size, k_size, s) in enumerate(zip(spatial_dims, kernel_shape, stride)):
                    pad_before, pad_after = AutoPad.compute_padding(
                        in_size, k_size, s, 1, auto_pad
                    )
                    pads.extend([pad_before, pad_after])
                
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
        stride = attrs.get('stride', kernel_size)
        padding = attrs.get('pads', 0)
        if isinstance(padding, list) and len(padding) > 0:
            # Convert ONNX pads format to PyTorch format
            if len(padding) == 4:  # 2D pool
                padding = (padding[1], padding[3], padding[0], padding[2])
            elif len(padding) == 2:  # 1D pool
                padding = (padding[0], padding[1])
            else:
                padding = tuple(padding)
        elif auto_pad != 'NOTSET':
            padding = 0  # Padding already handled by PadNode
        
        dilation = attrs.get('dilation', 1)
        ceil_mode = attrs.get('ceil_mode', False)
        
        pool_node = MaxPoolNode.create(
            name=node_name,
            inputs=pool_inputs,
            outputs=list(node_proto.output),
            input_tensors=pool_input_tensors,
            output_tensors=output_tensors,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode
        )
        nodes.append(pool_node)
        return nodes
    
    def _convert_average_pool(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """
        Convert ONNX AveragePool to AveragePoolNode.
        Handles AUTO_PAD by creating PadNode first if needed.
        Maps to AvgPool1d/2d/3d based on kernel_size.
        """
        nodes = []
        node_name = node_proto.name if node_proto.name else f"AveragePool_{node_index}"
        
        kernel_shape = attrs.get('kernel_size', attrs.get('kernel_shape', None))
        if kernel_shape is None:
            logger.warning(f"Could not infer kernel_shape for AveragePool {node_name}, defaulting to 2D")
            kernel_shape = (2, 2)
        
        if isinstance(kernel_shape, int):
            kernel_dims = 1
            kernel_size = kernel_shape
        else:
            kernel_dims = len(kernel_shape)
            kernel_size = kernel_shape[0] if len(kernel_shape) == 1 else kernel_shape
        
        # Handle AUTO_PAD
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        pad_node = None
        pool_inputs = list(node_proto.input)
        pool_input_tensors = input_tensors.copy()
        
        if auto_pad in ('SAME_UPPER', 'SAME_LOWER', 'VALID'):
            pad_name = f"{node_name}_pad"
            pad_output = f"{node_name}_padded"
            
            input_shape = input_tensors[pool_inputs[0]].shape
            if input_shape is None:
                logger.warning(f"Cannot compute auto_pad for AveragePool {node_name} with unknown input shape")
                auto_pad = 'NOTSET'
            else:
                stride = attrs.get('stride', kernel_size)
                if isinstance(stride, int):
                    stride = (stride,) * kernel_dims
                if isinstance(kernel_shape, int):
                    kernel_shape = (kernel_shape,)
                
                pads = []
                spatial_dims = input_shape[2:]
                for i, (in_size, k_size, s) in enumerate(zip(spatial_dims, kernel_shape, stride)):
                    pad_before, pad_after = AutoPad.compute_padding(
                        in_size, k_size, s, 1, auto_pad
                    )
                    pads.extend([pad_before, pad_after])
                
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
        
        stride = attrs.get('stride', kernel_size)
        padding = attrs.get('pads', 0)
        if isinstance(padding, list) and len(padding) > 0:
            if len(padding) == 4:
                padding = (padding[1], padding[3], padding[0], padding[2])
            elif len(padding) == 2:
                padding = (padding[0], padding[1])
            else:
                padding = tuple(padding)
        elif auto_pad != 'NOTSET':
            padding = 0
        
        ceil_mode = attrs.get('ceil_mode', False)
        count_include_pad = attrs.get('count_include_pad', True)
        
        pool_node = AveragePoolNode.create(
            name=node_name,
            inputs=pool_inputs,
            outputs=list(node_proto.output),
            input_tensors=pool_input_tensors,
            output_tensors=output_tensors,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )
        nodes.append(pool_node)
        return nodes
    
    def _convert_global_average_pool(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX GlobalAveragePool to GlobalAveragePoolNode."""
        node_name = node_proto.name if node_proto.name else f"GlobalAveragePool_{node_index}"
        return [GlobalAveragePoolNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors
        )]
    
    def _convert_batch_normalization(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """Convert ONNX BatchNormalization to BatchNormalizationNode."""
        node_name = node_proto.name if node_proto.name else f"BatchNormalization_{node_index}"
        eps = attrs.get('epsilon', 1e-5)
        momentum = attrs.get('momentum', 0.9)
        
        # ONNX BatchNorm takes 5 inputs: (X, scale, B, mean, var)
        # PyTorch/TIR BatchNorm also takes 5 inputs (same structure)
        return [BatchNormalizationNode.create(
            name=node_name,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            eps=eps,
            momentum=momentum
        )]
    
    def _convert_squeeze(self, node_proto, input_tensors, output_tensors, attrs, node_index):
        """
        Convert ONNX Squeeze to SqueezeNode.
        ONNX Squeeze (opset < 13): axes as attribute
        ONNX Squeeze (opset >= 13): axes as input tensor (second input)
        PyTorch/TIR Squeeze: 1 input, dim as attribute
        Forge Squeeze: 1 input, dim as int attribute (single dim only)
        """
        node_name = node_proto.name if node_proto.name else f"Squeeze_{node_index}"
        
        # Extract axes/dim from ONNX
        axes = attrs.get('axes', None)
        
        # If axes not in attrs, check if it's provided as input tensor (opset >= 13)
        if axes is None and len(node_proto.input) > 1:
            axes_input = node_proto.input[1]
            # Try to get axes from initializers
            if hasattr(self, 'graph_proto') and self.graph_proto:
                for init in self.graph_proto.initializer:
                    if init.name == axes_input:
                        from onnx import numpy_helper
                        axes_array = numpy_helper.to_array(init)
                        axes = tuple(int(x) for x in axes_array.tolist())
                        break
        
        # Convert to dim (Forge only supports single dim, but PyTorch supports tuple)
        if axes is not None:
            if isinstance(axes, (list, tuple)):
                if len(axes) == 1:
                    dim = axes[0]
                else:
                    # Multiple dims - Forge only supports one, but we'll store as tuple for PyTorch compatibility
                    # In emit(), we'll use the first one
                    dim = tuple(axes)
            else:
                dim = axes
        else:
            # No axes specified - squeeze all dims of size 1
            # Forge requires dim, so we'll need to infer from input shape
            input_info = list(input_tensors.values())[0]
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
        
        # Create TIR node with only data input
        data_input = node_proto.input[0]
        tir_input_tensors = {data_input: input_tensors[data_input]}
        
        return [SqueezeNode.create(
            name=node_name,
            inputs=[data_input],  # Only data input
            outputs=list(node_proto.output),
            input_tensors=tir_input_tensors,
            output_tensors=output_tensors,
            dim=dim
        )]

    def transpile(self, onnx_model: onnx.ModelProto) -> TIRGraph:
        """Transpile an ONNX model to a TIR graph."""
        logger.info("Starting Transpilation with Shape Inference...")
        
        # Store original model for debug mode and converter access
        self.onnx_model = onnx_model
        
        # 1. Perform Shape Inference
        try:
            inferred_model = shape_inference.infer_shapes(onnx_model)
        except Exception as e:
            logger.error(f"Shape inference failed: {e}. Proceeding without inferred shapes.")
            inferred_model = onnx_model

        # 2. Remove Initializers from Graph Inputs
        inferred_model = remove_initializers_from_input(inferred_model)

        graph_proto = inferred_model.graph
        # Store graph_proto for converter methods to access initializers
        self.graph_proto = graph_proto
        tir_graph = TIRGraph(name=graph_proto.name, frontend_model=onnx_model if self.debug else None, debug_mode=self.debug)

        # 3. Create map of all value infos (including model inputs/outputs)
        # This map contains all tensor names mapped to their full ONNX metadata (shape, type)
        value_info_map = {vi.name: vi for vi in graph_proto.value_info}
        value_info_map.update({vi.name: vi for vi in graph_proto.input})
        value_info_map.update({vi.name: vi for vi in graph_proto.output})

        # 4. Process Initializers (Weights/Parameters)
        for initializer in graph_proto.initializer:
            np_array = numpy_helper.to_array(initializer)
            
            # Use the new utility to correctly determine PyTorch dtype
            onnx_dtype = initializer.data_type
            torch_dtype = onnx_dtype_to_torch_dtype(onnx_dtype)
            
            torch_tensor = torch.from_numpy(np_array).to(torch_dtype)
            tir_graph.initializers[initializer.name] = torch_tensor

        # 5. Process remaining Graph Inputs (Actual model inputs)
        # Use utility function to get input names
        tir_graph.inputs = get_inputs_names(graph_proto)

        # 6. Process Nodes
        for i, node_proto in enumerate(graph_proto.node):
            op_type = node_proto.op_type
            
            # --- Input Tensor Metadata: Now uses TensorInfo ---
            input_tensors = {}
            for name in node_proto.input:
                input_tensors[name] = self._get_tensor_info(value_info_map, name)

            # --- Output Tensor Metadata: Now uses TensorInfo ---
            output_tensors = {}
            for name in node_proto.output:
                output_tensors[name] = self._get_tensor_info(value_info_map, name)

            # Use enhanced attribute extraction
            attrs = extract_attributes(node_proto)

            # Get converter method for this op type
            converter_method = self._op_converters.get(op_type, None)
            
            if converter_method:
                # Use converter method to create TIR node(s)
                tir_nodes = converter_method(node_proto, input_tensors, output_tensors, attrs, i)
                
                # Add all nodes returned by converter (may be multiple for multi-node conversions)
                for tir_node in tir_nodes:
                    tir_graph.add_node(tir_node)
                    
                    # Store ONNX node proto for debug mode
                    if self.debug:
                        # For multi-node conversions, map all TIR nodes to the original ONNX node
                        tir_graph.node_proto_map[tir_node.name] = node_proto
            else:
                # Fallback to GenericNode for unsupported operations
                node_name = node_proto.name if node_proto.name else f"{op_type}_{i}"
                tir_node = GenericNode(
                    name=node_name,
                    op_type=op_type,
                    inputs=list(node_proto.input),
                    outputs=list(node_proto.output),
                    input_tensors=input_tensors,
                    output_tensors=output_tensors,
                    attrs=attrs
                )
                tir_graph.add_node(tir_node)
                
                # Store ONNX node proto for debug mode
                if self.debug:
                    tir_graph.node_proto_map[node_name] = node_proto

        # 7. Process Graph Outputs
        # Use utility function to get output names
        tir_graph.outputs = get_outputs_names(graph_proto)

        # 8. Compute activation dependencies for memory management
        tir_graph.compute_activation_dependencies()
        
        # Debug mode is already set in TIRGraph constructor

        return tir_graph

