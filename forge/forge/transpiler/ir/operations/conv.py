"""
Convolution operations: Conv1d, Conv2d, Conv3d
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple

from ..nodes import TIRNode
from ..types import TensorInfo



class Conv1dNode(TIRNode):
    """
    PyTorch-like Conv1d operation.
    Note: Forge does not have Conv1d operation - only Conv2d is supported.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               stride: Union[int, Tuple[int]] = 1,
               padding: Union[int, str, Tuple[int]] = 0,
               dilation: Union[int, Tuple[int]] = 1,
               groups: int = 1) -> 'Conv1dNode':
        """Static factory method to create a Conv1dNode."""
        raise NotImplementedError(
            "Conv1d is not supported in Forge. Only Conv2d operation is available. "
            "Please convert 1D convolution to 2D by adding a spatial dimension."
        )



class Conv2dNode(TIRNode):
    """
    PyTorch-like Conv2d operation.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               stride: Union[int, Tuple[int, int]] = 1,
               padding: Union[int, str, Tuple[int, int]] = 0,
               dilation: Union[int, Tuple[int, int]] = 1,
               groups: int = 1) -> 'Conv2dNode':
        """Static factory method to create a Conv2dNode."""
        return Conv2dNode(
            name=name,
            op_type="Conv2d",
            inputs=inputs,
            outputs=outputs,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            attrs={
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'groups': groups
            },
            forge_op_function_name="forge.op.convolution.Conv2d"
        )

    def eval(self, input_tensors):
        x = input_tensors[self.inputs[0]]
        w = input_tensors[self.inputs[1]]
        b = input_tensors[self.inputs[2]] if len(self.inputs) > 2 else None
        return {self.outputs[0]: F.conv2d(x, w, bias=b, **self.attrs)}



class Conv3dNode(TIRNode):
    """
    PyTorch-like Conv3d operation.
    Note: Forge does not have Conv3d operation - only Conv2d is supported.
    """
    @staticmethod
    def create(name: str, inputs: List[str], outputs: List[str],
               input_tensors: Dict[str, TensorInfo],
               output_tensors: Dict[str, TensorInfo],
               stride: Union[int, Tuple[int, int, int]] = 1,
               padding: Union[int, str, Tuple[int, int, int]] = 0,
               dilation: Union[int, Tuple[int, int, int]] = 1,
               groups: int = 1) -> 'Conv3dNode':
        """Static factory method to create a Conv3dNode."""
        raise NotImplementedError(
            "Conv3d is not supported in Forge. Only Conv2d operation is available. "
            "Please convert 3D convolution to 2D by reshaping or using multiple 2D convolutions."
        )

