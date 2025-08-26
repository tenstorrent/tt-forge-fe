# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Tuple, List

from ..tensor import Tensor
from ..parameter import Parameter
from .common import ForgeOp as op


def conv2d_padding_to_canonical(padding, kernel_size):
    # current implementation is without dilation

    assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), "Unsupported kernel size"
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

    if isinstance(padding, int):
        return [padding] * 4
    elif isinstance(padding, str):
        assert padding == "same", "Unsupported padding"
        padding = [kW // 2] * 2 + [kH // 2] * 2
        if kW % 2 == 0:
            padding[1] -= 1
        if kH % 2 == 0:
            padding[3] -= 1
        return padding
    elif isinstance(padding, tuple) or isinstance(padding, list):
        if len(padding) == 2:
            return [padding[1]] * 2 + [padding[0]] * 2
        elif len(padding) == 4:
            return list(padding)
        else:
            raise AssertionError("Unsupported padding")
    else:
        raise AssertionError("Unsupported padding")


def Conv2d(
    name: str,
    activations: Tensor,
    weights: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]] = None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, str, List[int]] = "same",
    dilation: Union[int, List[int]] = 1,
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
        stride=stride,  # [sH, sW]
        dilation=dilation,  # [dH, dW]
        groups=groups,
        padding=padding,  # [pT, pL, pB, pR]
        channel_last=channel_last,
    ).get_tensor()


def Conv2dTranspose(
    name: str,
    activations: Tensor,
    weights: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]] = None,
    stride: int = 1,
    padding: Union[int, str, Tuple[int, int, int, int]] = "same",
    dilation: int = 1,
    groups: int = 1,
    channel_last: bool = False,
    output_padding: Union[int, Tuple[int, int]] = 0,
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
        padding = [padding] * 2

    if isinstance(padding, tuple):
        # padding is tuple (top, left, bottom, right)
        top, left, bottom, right = padding
        assert (
            left == right and top == bottom
        ), "padding must be a tuple of (top, left, bottom, right) where left == right and top == bottom"
        padding = [top, left]

    if isinstance(dilation, int):
        dilation = [dilation] * 2

    if isinstance(output_padding, int):
        output_padding = [output_padding] * 2

    inputs = [activations, weights]
    if bias is not None:
        inputs.append(bias)

    return op(
        "conv2d_transpose",
        name,
        *inputs,
        stride=stride,  # [sH, sW]
        dilation=dilation,  # [dH, dW]
        groups=groups,
        padding=padding,  # [pH, pW]
        output_padding=output_padding,  # [opH, opW]
        channel_last=channel_last,
    ).get_tensor()
