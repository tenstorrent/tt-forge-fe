"""
Pooling operations: MaxPool, AveragePool, GlobalAveragePool
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple

from ..nodes import TIRNode
from ..types import TensorInfo
from ...frontends.onnx.converters.autopad import AutoPad



class MaxPoolNode(TIRNode):
    """
    PyTorch-like MaxPool operation.
    Supports 1D, 2D, and 3D pooling based on kernel_size.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               kernel_size: Union[int, Tuple[int, ...]],
               stride: Union[int, Tuple[int, ...]] = None,
               padding: Union[int, Tuple[int, ...]] = 0,
               dilation: Union[int, Tuple[int, ...]] = 1,
               ceil_mode: bool = False) -> 'MaxPoolNode':
        """Static factory method to create a MaxPoolNode."""
        if stride is None:
            stride = kernel_size
        
        # Determine function name based on kernel_size
        if isinstance(kernel_size, int):
            ndim = 1
            func_name = "forge.op.MaxPool1d"
        elif isinstance(kernel_size, (list, tuple)):
            ndim = len(kernel_size)
            if ndim == 1:
                func_name = "forge.op.MaxPool1d"
            elif ndim == 2:
                func_name = "forge.op.MaxPool2d"
            else:
                func_name = "forge.op.MaxPool2d"  # Default to 2D
        else:
            func_name = "forge.op.MaxPool2d"  # Default
        
        return MaxPoolNode(
            name=name,
            op_type="MaxPool",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'ceil_mode': ceil_mode
            },
            forge_op_function_name=func_name
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        kernel_size = self.attrs['kernel_size']
        stride = self.attrs.get('stride', kernel_size)
        padding = self.attrs.get('padding', 0)
        dilation = self.attrs.get('dilation', 1)
        ceil_mode = self.attrs.get('ceil_mode', False)
        
        # Determine pooling dimension from kernel_size
        if isinstance(kernel_size, int):
            ndim = 1
        else:
            ndim = len(kernel_size)
        
        if ndim == 1:
            return {self.outputs[0]: F.max_pool1d(x, kernel_size, stride, padding, dilation, ceil_mode)}
        elif ndim == 2:
            return {self.outputs[0]: F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)}
        elif ndim == 3:
            return {self.outputs[0]: F.max_pool3d(x, kernel_size, stride, padding, dilation, ceil_mode)}
        else:
            raise ValueError(f"Unsupported MaxPool dimension: {ndim}")



class AveragePoolNode(TIRNode):
    """
    PyTorch-like AveragePool operation.
    Supports 1D, 2D, and 3D pooling based on kernel_size.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               kernel_size: Union[int, Tuple[int, ...]],
               stride: Union[int, Tuple[int, ...]] = None,
               padding: Union[int, Tuple[int, ...]] = 0,
               ceil_mode: bool = False,
               count_include_pad: bool = True) -> 'AveragePoolNode':
        """Static factory method to create an AveragePoolNode."""
        if stride is None:
            stride = kernel_size
        
        # Determine function name based on kernel_size
        if isinstance(kernel_size, int):
            ndim = 1
            func_name = "forge.op.AvgPool1d"
        elif isinstance(kernel_size, (list, tuple)):
            ndim = len(kernel_size)
            if ndim == 1:
                func_name = "forge.op.AvgPool1d"
            elif ndim == 2:
                func_name = "forge.op.AvgPool2d"
            else:
                func_name = "forge.op.AvgPool2d"  # Default to 2D
        else:
            func_name = "forge.op.AvgPool2d"  # Default
        
        return AveragePoolNode(
            name=name,
            op_type="AveragePool",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'ceil_mode': ceil_mode,
                'count_include_pad': count_include_pad
            },
            forge_op_function_name=func_name
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        kernel_size = self.attrs['kernel_size']
        stride = self.attrs.get('stride', kernel_size)
        padding = self.attrs.get('padding', 0)
        ceil_mode = self.attrs.get('ceil_mode', False)
        count_include_pad = self.attrs.get('count_include_pad', True)
        
        # Determine pooling dimension from kernel_size
        if isinstance(kernel_size, int):
            ndim = 1
        else:
            ndim = len(kernel_size)
        
        if ndim == 1:
            return {self.outputs[0]: F.avg_pool1d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}
        elif ndim == 2:
            return {self.outputs[0]: F.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}
        elif ndim == 3:
            return {self.outputs[0]: F.avg_pool3d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)}
        else:
            raise ValueError(f"Unsupported AveragePool dimension: {ndim}")



class GlobalAveragePoolNode(TIRNode):
    """
    PyTorch-like GlobalAveragePool operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo]) -> 'GlobalAveragePoolNode':
        """Static factory method to create a GlobalAveragePoolNode."""
        return GlobalAveragePoolNode(
            name=name,
            op_type="GlobalAveragePool",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={},
            forge_op_function_name="forge.op.ReduceAvg"  # GlobalAveragePool can be represented as ReduceAvg
        )
    
    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        # Global average pool over spatial dimensions (H, W)
        return {self.outputs[0]: F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)}

