# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List

from ..tensor import Tensor
from ..parameter import Parameter
from .common import ForgeOp as op

RESIZE2d_METHOD_TO_INT = {
    "nearest_neighbor": 0,
    "linear": 1,
    "bilinear": 1,
    "cubic": 2,
}

INT_TO_RESIZE2d_METHOD = {
    0: "nearest",
    1: "bilinear",
    2: "cubic",
}


def Resize2d(
    name: str,
    operandA: Tensor,
    sizes: List[int],
    method: str = "nearest_neighbor",
    align_corners=False,
    extrapolation_value: int = 0,
    channel_last: bool = False,
) -> Tensor:
    """
    Resize input activations, with default method 'nearest_neighbor'

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    sizes: List[int]
        The target 2D sizes to extrapolate to

    method: str
        Extrapolation method

    extrapolation_value: int

    """
    assert len(sizes) == 2
    assert (
        method == "nearest_neighbor" or method == "linear" or method == "bilinear" or method == "cubic"
    ), "Only support nearest_neighbor, linear and cubic interpolation for now"

    if isinstance(channel_last, int):
        channel_last = bool(channel_last)

    result: Tensor = op(
        "resize2d",
        name,
        operandA,
        attrs=(*sizes, RESIZE2d_METHOD_TO_INT[method], int(align_corners), int(channel_last)),
        sizes=sizes,
        method=RESIZE2d_METHOD_TO_INT[method],
        align_corners=align_corners,
        channel_last=channel_last,
    ).get_tensor()

    return result


def Upsample2d(
    name: str, operandA: Tensor, scale_factor: int, mode: str = "nearest", channel_last: bool = False
) -> Tensor:
    """
    Upsample 2D operation

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    scale_factor: int
        multiplier for spatial size.

    mode: str
        the upsampling algorithm

    Returns
    -------
    Tensor
        Forge tensor
    """
    result: Tensor = op(
        "upsample2d",
        name,
        operandA,
        attrs=(scale_factor, mode, channel_last),
        scale_factor=scale_factor,
        mode=mode,
        channel_last=channel_last,
    ).get_tensor()

    return result


def Downsample2d(
    name: str, operandA: Tensor, scale_factor: int, mode: str = "nearest", channel_last: bool = False
) -> Tensor:
    """
    Downsample 2D operation

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    scale_factor: int
        Divider for spatial size.

    mode: str
        The downsampling algorithm

    channel_last: bool
        Whether the input is in channel-last format (NHWC)

    Returns
    -------
    Tensor
        Forge tensor
    """
    result: Tensor = op(
        "downsample2d",
        name,
        operandA,
        attrs=(scale_factor, mode, channel_last),
        scale_factor=scale_factor,
        mode=mode,
        channel_last=channel_last,
    ).get_tensor()

    return result


def Resize3d(
    name: str,
    operandA: Tensor,
    sizes: List[int],
    method: str = "nearest_neighbor",
    align_corners=False,
    extrapolation_value: int = 0,
    channel_last: bool = False,
) -> Tensor:
    """
    Resize input activations, with default method 'nearest_neighbor'

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    sizes: List[int]
        The target 2D sizes to extrapolate to

    method: str
        Extrapolation method

    extrapolation_value: int

    """
    assert len(sizes) == 3
    assert method == "nearest_neighbor", "Only support nearest_neighbor for now"
    assert not channel_last, "Decomposition for channel-last Resize3d is not added yet"
    result: Tensor = op(
        "resize3d",
        name,
        operandA,
        attrs=(*sizes, RESIZE2d_METHOD_TO_INT[method], int(align_corners), int(channel_last)),
    ).get_tensor()

    return result
