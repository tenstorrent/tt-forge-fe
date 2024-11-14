# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Tuple, List

from ..tensor import Tensor
from ..parameter import Parameter
from .common import ForgeOp as op

from forge.op.eval.sparse_utils import conv2d_padding_to_canonical, conv3d_padding_to_canonical


def Conv2d(
    name: str,
    activations: Tensor,
    weights: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]] = None,
    stride: int = 1,
    padding: Union[int, str, List] = "same",
    dilation: Union[int, List] = 1,
    groups: int = 1,
    channel_last: bool = False,
) -> Tensor:
    """
    Conv2d transformation on input activations, with optional bias.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iH, iW)

    weights:
        Tensor
            Input weights of shape (Cout, Cin / groups, kH, kW)
        [Tensor]
            Internal Use pre-split
            Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)]
            of length: (K*K // weight_grouping)

    bias: Tenor, optional
        Optional bias tensor of shape (Cout)
    """
    if isinstance(stride, int):
        stride = [stride] * 2
    if isinstance(dilation, int):
        dilation = [dilation] * 2

    padding = conv2d_padding_to_canonical(padding, (weights.shape[2], weights.shape[3]))

    inputs = [activations, weights]
    if bias is not None:
        inputs.append(bias)

    return op(
        "conv2d",
        name,
        *inputs,
        stride_height=stride[0],
        stride_width=stride[1],
        dilation_height=dilation[0],
        dilation_width=dilation[1],
        groups=groups,
        padding_left=padding[0],
        padding_right=padding[1],
        padding_top=padding[2],
        padding_bottom=padding[3],
        channel_last=channel_last,
    ).get_tensor()


def Conv2dTranspose(
    name: str,
    activations: Tensor,
    weights: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]] = None,
    stride: int = 1,
    padding: Union[int, str] = "same",
    dilation: int = 1,
    groups: int = 1,
    channel_last: bool = False,
) -> Tensor:
    """
    Conv2dTranspose transformation on input activations, with optional bias.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iH, iW)

    weights:
        Tensor
            Input weights of shape (Cout, Cin / groups, kH, kW)
        [Tensor]
            Internal Use pre-split
            Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)]
            of length: (K*K // weight_grouping)

    bias: Tenor, optional
        Optional bias tensor of shape (Cout)
    """
    if isinstance(stride, int):
        stride = [stride] * 2

    if padding == "same":
        padding = [weights.shape[3] // 2] * 2 + [weights.shape[2] // 2] * 2

    if isinstance(padding, int):
        padding = [padding] * 4  # [left, right, top, bottom]

    inputs = [activations, weights]
    if bias is not None:
        inputs.append(bias)

    # Attrs are:
    # [
    #     stride_height,
    #     stride_width,
    #     dilation,
    #     groups,
    #     padding_left,
    #     padding_right,
    #     padding_top,
    #     padding_bottom,
    #     channel_last,
    # ]
    attrs = stride + [dilation, groups] + padding + [channel_last]
    return op(
        "conv2d_transpose",
        name,
        *inputs,
        attrs=attrs,
        stride_height=stride[0],
        stride_width=stride[1],
        dilation_height=dilation,
        dilation_width=dilation,
        groups=groups,
        padding_left=padding[0],
        padding_right=padding[1],
        padding_top=padding[2],
        padding_bottom=padding[3],
        channel_last=channel_last,
    ).get_tensor()


def Conv3d(
    name: str,
    activations: Tensor,
    weights: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]] = None,
    stride: int = 1,
    padding: Union[int, str, List] = "same",
    dilation: Union[int, List] = 1,
    groups: int = 1,
    channel_last: bool = False,
) -> Tensor:
    """
    Conv3d on input activations, with optional bias.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iD, iH, iW) if channel_last=False,
        or (N, iD, iH, iW, Cin) if channel_last=True

    weights:
        Tensor
            Input weights of shape (Cout, Cin / groups, kD, kH, kW)

    bias: Tensor, optional
        Optional bias tensor of shape (Cout)
    """
    # Ensure stride, dilation, and padding are in the correct format
    if isinstance(stride, int):
        stride = [stride] * 3
    if isinstance(dilation, int):
        dilation = [dilation] * 3

    # Adjust padding to handle 3D dimensions
    padding = conv3d_padding_to_canonical(padding, (weights.shape[0], weights.shape[3], weights.shape[4]))

    # Assemble inputs list
    inputs = [activations, weights]
    if bias is not None:
        inputs.append(bias)

    return op(
        "conv3d",
        name,
        *inputs,
        stride_depth=stride[0],
        stride_height=stride[1],
        stride_width=stride[2],
        dilation_depth=dilation[0],
        dilation_height=dilation[1],
        dilation_width=dilation[2],
        groups=groups,
        padding_front=padding[0],
        padding_back=padding[1],
        padding_top=padding[2],
        padding_bottom=padding[3],
        padding_left=padding[4],
        padding_right=padding[5],
        channel_last=channel_last,
    ).get_tensor()
