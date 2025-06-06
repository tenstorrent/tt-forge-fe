# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Tuple, List

from ..tensor import Tensor
from ..parameter import Parameter
from .common import ForgeOp as op


def MaxPool1d(
    name: str,
    activations: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: int = 1,
    padding: Union[int, str] = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    """
    MaxPool1d transformation on input activations

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iW)

    kernel_size:
        Size of pooling region
    """
    assert ceil_mode == False, f"Unsupported arg: ceil_mode = {ceil_mode}"
    assert return_indices == False, f"Unsupported arg: return_indices = {return_indices}"
    assert type(kernel_size) is int, f"Unsupported arg type: type of kernel_size ({type(kernel_size)}) != int"
    assert type(padding) is int, f"Unsupported arg type: type of padding ({type(padding)}) != int"

    attrs = [kernel_size, stride, dilation, ceil_mode, padding]
    return op(
        "max_pool1d",
        name,
        activations,
        attrs=attrs,
    ).get_tensor()


def MaxPool2d(
    name: str,
    activations: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: int = 1,
    padding: Union[int, str] = "same",
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
    max_pool_add_sub_surround: bool = False,
    max_pool_add_sub_surround_value: float = 1.0,
    channel_last: bool = False,
) -> Tensor:
    """
    Maxpool2d transformation on input activations

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iH, iW)

    kernel_size:
        Size of pooling region
    """
    assert return_indices == False, f"Unsupported arg: return_indices = {return_indices}"
    assert isinstance(kernel_size, int) or (
        isinstance(kernel_size, Tuple) and all(isinstance(k_dim, int) for k_dim in kernel_size)
    ), "Unsupported"
    if isinstance(stride, int):
        stride = [stride] * 2

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 2
    elif isinstance(kernel_size, Tuple):
        kernel_size = list(kernel_size)
    if padding == "same":
        padding = [kernel_size[1] // 2] * 2 + [kernel_size[0] // 2] * 2
    if isinstance(padding, int):
        padding = [padding] * 4  # [left,right,top,bottom]

    attrs = (
        kernel_size
        + stride
        + [dilation, ceil_mode]
        + padding
        + [max_pool_add_sub_surround, max_pool_add_sub_surround_value]
        + [channel_last]
    )
    return op(
        "max_pool2d",
        name,
        activations,
        kernel_height=kernel_size[0],
        kernel_width=kernel_size[1],
        stride_height=stride[0],
        stride_width=stride[1],
        dilation_height=dilation,
        dilation_width=dilation,
        ceil_mode=ceil_mode,
        padding_left=padding[0],
        padding_right=padding[1],
        padding_top=padding[2],
        padding_bottom=padding[3],
        channel_last=channel_last,
    ).get_tensor()


def MaxPool3d(
    name: str,
    activations: Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: int = 1,
    padding: Union[int, str] = "same",
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
    max_pool_add_sub_surround: bool = False,
    max_pool_add_sub_surround_value: float = 1.0,
    channel_last: bool = False,
) -> Tensor:
    """
    Maxpool3d transformation on input activations

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iH, iW)

    kernel_size:
        Size of pooling region
    """
    assert not channel_last, "Decomposition for channel-last MaxPool3d is not added yet"
    assert return_indices == False, f"Unsupported arg: return_indices = {return_indices}"
    assert isinstance(kernel_size, int) or (
        isinstance(kernel_size, Tuple) and all(isinstance(k_dim, int) for k_dim in kernel_size)
    ), "Unsupported"
    if isinstance(stride, int):
        stride = [stride] * 3

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 3
    elif isinstance(kernel_size, Tuple):
        kernel_size = list(kernel_size)
    if padding == "same":
        padding = [kernel_size[2] // 2] * 2 + [kernel_size[1] // 2] * 2 + [kernel_size[0] // 2] * 2
    if isinstance(padding, int):
        padding = [padding] * 6  # [left,right,top,bottom, depth_first, depth_last]

    attrs = (
        kernel_size
        + stride
        + [dilation, ceil_mode]
        + padding
        + [max_pool_add_sub_surround, max_pool_add_sub_surround_value]
        + [channel_last]
    )
    return op(
        "max_pool3d",
        name,
        activations,
        attrs=attrs,
    ).get_tensor()


def AvgPool1d(
    name: str,
    activations: Tensor,
    kernel_size: Union[
        int,
        Tuple[
            int,
        ],
    ],
    stride: int = 1,
    padding: Union[int, str] = "same",
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """
    Avgpool1d transformation on input activations

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iW)

    kernel_size:
        Size of pooling region
    """

    assert isinstance(kernel_size, (int, tuple, list)), "Unsupported"

    if isinstance(stride, int):
        stride = [stride]

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]
    elif isinstance(kernel_size, Tuple):
        kernel_size = list(kernel_size)

    if padding == "same":
        padding = [kernel_size[1] // 2] + [kernel_size[0] // 2]
    if isinstance(padding, int):
        padding = [padding] * 2  # [left,right]

    dilation = 1  # Only as place holder to standardize interface with MaxPool2d
    attrs = kernel_size + stride + [dilation, ceil_mode] + padding + [count_include_pad]
    return op(
        "avg_pool1d",
        name,
        activations,
        attrs=attrs,  # 1 is placeholder for dilation
    ).get_tensor()


def AvgPool2d(
    name: str,
    activations: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: int = 1,
    padding: Union[int, str] = "same",
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: float = None,
    channel_last: bool = False,
) -> Tensor:
    """
    Avgpool2d transformation on input activations

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iH, iW)

    kernel_size:
        Size of pooling region
    """
    assert divisor_override is None, "Unsupported"
    assert isinstance(kernel_size, (int, tuple, list)), "Unsupported"

    if isinstance(stride, int):
        stride = [stride] * 2

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 2
    elif isinstance(kernel_size, Tuple):
        kernel_size = list(kernel_size)
    if padding == "same":
        padding = [kernel_size[1] // 2] * 2 + [kernel_size[0] // 2] * 2
    if isinstance(padding, int):
        padding = [padding] * 4  # [left,right,top,bottom]

    dilation = 1  # Only as place holder to standardize interface with MaxPool2d
    attrs = kernel_size + stride + [dilation, ceil_mode] + padding + [count_include_pad] + [channel_last]
    return op(
        "avg_pool2d",
        name,
        activations,
        attrs=attrs,  # 1 is placeholder for dilation
    ).get_tensor()


def AdaptiveMaxPool2d(
    name: str,
    activations: Tensor,
    output_size: Union[int, Tuple[int, int]],
    channel_last: bool = False,
) -> Tensor:
    """
    Adaptive MaxPool2d transformation on input activations

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset
    activations: Tensor
        Input tensor of shape (N, C, H, W)
    output_size: int or Tuple[int, int]
        The target output size (height, width) after pooling
    channel_last: bool
        Whether the input tensor has channel last layout (NHWC). Default is False (NCHW)
    """

    if isinstance(output_size, int):
        output_size = [output_size, output_size]
    elif isinstance(output_size, tuple):
        output_size = list(output_size)

    assert len(output_size) == 2, f"Expected 2D output size, got: {output_size}"

    attrs = output_size + [channel_last]

    return op(
        "adaptive_max_pool2d",
        name,
        activations,
        attrs=attrs,
    ).get_tensor()


def AvgPool3d(
    name: str,
    activations: Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: int = 1,
    padding: Union[int, str] = "same",
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: float = None,
    channel_last: bool = False,
) -> Tensor:
    """
    Avgpool3d transformation on input activations
    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset
    activations: Tensor
        Input activations of shape (N, Cin, iD, iH, iW)
    kernel_size:
        Size of pooling region
    """
    assert divisor_override is None, f"divisor_override={divisor_override} is not supported. Please set it to None."
    assert isinstance(
        kernel_size, (int, tuple, list)
    ), f"Invalid type for kernel_size: {type(kernel_size)}. Expected an int or tuple/list of integers"

    if isinstance(stride, int):
        stride = [stride] * 3

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 3
    elif isinstance(kernel_size, Tuple):
        kernel_size = list(kernel_size)
    if padding == "same":

        padding = [
            kernel_size[2] // 2,
            kernel_size[2] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[0] // 2,
            kernel_size[0] // 2,
        ]
    if isinstance(padding, int):
        padding = [padding] * 6  # [left, right, top, bottom, front, back]

    dilation = 1  # Only as place holder to standardize interface with MaxPool3d
    attrs = kernel_size + stride + [dilation, ceil_mode] + padding + [count_include_pad] + [channel_last]

    return op(
        "avg_pool3d",
        name,
        activations,
        attrs=attrs,  # 1 is placeholder for dilation
    ).get_tensor()
