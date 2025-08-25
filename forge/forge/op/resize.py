# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple, Union

from ..tensor import Tensor
from .common import ForgeOp as op


def Resize1d(
    name: str,
    operandA: Tensor,
    size: int,
    mode: str = "nearest",
    align_corners: bool = False,
    channel_last: bool = False,
) -> Tensor:
    """
    Resize input activations, with default mode 'nearest'

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    size: int
        The target size to extrapolate

    mode: str
        Interpolation mode

    channel_last: bool
        Whether the input is in channel-last format (NWC)

    """

    assert isinstance(size, int)
    assert mode in ["nearest", "linear"], "Only support nearest and linear mode for now"

    result: Tensor = op(
        "resize1d",
        name,
        operandA,
        size=size,
        mode=mode,
        align_corners=align_corners,
        channel_last=channel_last,
    ).get_tensor()

    return result


def Resize2d(
    name: str,
    operandA: Tensor,
    sizes: Union[List[int], Tuple[int, int]],
    mode: str = "nearest",
    align_corners: bool = False,
    channel_last: bool = False,
) -> Tensor:
    """
    Resize input activations, with default mode 'nearest'

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    sizes: Union[List[int], Tuple[int, int]]
        The target 2D sizes to extrapolate to

    mode: str
        Interpolation mode

    channel_last: bool
        Whether the input is in channel-last format (NHWC)

    """

    assert isinstance(sizes, (list, tuple)) and len(sizes) == 2
    assert mode in ["nearest", "bilinear"], "Only support nearest and bilinear mode for now"

    result: Tensor = op(
        "resize2d",
        name,
        operandA,
        sizes=sizes,
        mode=mode,
        align_corners=align_corners,
        channel_last=channel_last,
    ).get_tensor()

    return result


def Upsample2d(
    name: str,
    operandA: Tensor,
    scale_factor: Union[int, List[int], Tuple[int, int]],
    mode: str = "nearest",
    channel_last: bool = False,
) -> Tensor:
    """
    Upsample 2D operation

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    scale_factor: Union[int, List[int], Tuple[int, int]]
        multiplier for spatial size.

    mode: str
        the upsampling algorithm

    Returns
    -------
    Tensor
        Forge tensor
    """

    assert (isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 2) or isinstance(
        scale_factor, int
    ), f"Only support List/Tuple of int (or) int scale_factor but provided {type(scale_factor)}"
    assert mode in ["nearest", "bilinear"], "Only support nearest and bilinear interpolation for now"

    if isinstance(scale_factor, int):
        scale_factor = (scale_factor, scale_factor)

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
    name: str,
    operandA: Tensor,
    scale_factor: Union[int, List[int], Tuple[int, int]],
    mode: str = "nearest",
    channel_last: bool = False,
) -> Tensor:
    """
    Downsample 2D operation

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    scale_factor: Union[int, List[int], Tuple[int, int]]
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

    assert (isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 2) or isinstance(
        scale_factor, int
    ), f"Only support List/Tuple of int (or) int scale_factor but provided {type(scale_factor)}"
    assert mode in ["nearest", "bilinear"], "Only support nearest and bilinear interpolation for now"

    if isinstance(scale_factor, int):
        scale_factor = (scale_factor, scale_factor)

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
