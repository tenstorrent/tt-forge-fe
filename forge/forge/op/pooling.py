# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union, Tuple

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

    return op(
        "max_pool1d",
        name,
        activations,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        ceil_mode=ceil_mode,
        padding=padding,
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
    if isinstance(channel_last, int):
        channel_last = bool(channel_last)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # change padding from [pL, pR, pT, pB] to [pT, pL, pB, pR]
    padding = [padding[2], padding[0], padding[3], padding[1]]

    return op(
        "max_pool2d",
        name,
        activations,
        kernel=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        ceil_mode=ceil_mode,
        channel_last=channel_last,
    ).get_tensor()


def AvgPool1d(
    name: str,
    activations: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
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
    return op(
        "avg_pool1d",
        name,
        activations,
        kernel_size=kernel_size[0],
        stride=stride[0],
        dilation=dilation,
        ceil_mode=ceil_mode,
        padding_left=padding[0],
        padding_right=padding[1],
        count_include_pad=count_include_pad,
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
    if isinstance(channel_last, int):
        channel_last = bool(channel_last)

    # change padding from [pL, pR, pT, pB] to [pT, pL, pB, pR]
    padding = [padding[2], padding[0], padding[3], padding[1]]

    # Only as place holder to standardize interface with MaxPool2d
    dilation = (1, 1)

    return op(
        "avg_pool2d",
        name,
        activations,
        kernel=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        channel_last=channel_last,
    ).get_tensor()
