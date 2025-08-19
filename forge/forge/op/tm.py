# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union, Tuple, List
from ..forgeglobal import TILE_DIM
from .common import ForgeOp as op
from ..tensor import Tensor, pytorch_dtype_to_forge_dataformat

import torch


def Transpose(name: str, operandA: Tensor, dim0: int, dim1: int) -> Tensor:
    """
    Tranpose X and Y (i.e. rows and columns) dimensions.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Forge tensor
    """
    assert dim0 != dim1

    dims = len(operandA.shape.dims)
    if dim0 >= 0:
        dim0 -= dims

    if dim1 >= 0:
        dim1 -= dims

    assert dim0 < 0
    assert dim1 < 0

    if dim0 > dim1:
        dim0, dim1 = dim1, dim0

    return op("transpose", name, operandA, attrs=(dim0, dim1), dim0=dim0, dim1=dim1).get_tensor()


def Reshape(name: str, operandA: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    Returns
    -------
    Tensor
        Forge tensor
    """
    tensor_volume = 1
    for dim in operandA.shape.dims:
        tensor_volume *= dim

    blank_idx = -1
    volume = 1
    for idx, d in enumerate(shape):
        if d == -1:
            assert blank_idx == -1, "Cannot have multiple -1 dims"
            blank_idx = idx
        else:
            volume *= d

    if blank_idx != -1:
        assert (tensor_volume % volume) == 0, "-1 dim does not divide evenly"
        shape[blank_idx] = tensor_volume // volume
        volume *= shape[blank_idx]

    assert tensor_volume == volume

    return op("reshape", name, operandA, shape=shape).get_tensor()


def Index(name: str, operandA: Tensor, dim: int, start: int, stop: int = None, stride: int = 1) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to slice

    start: int
        Starting slice index (inclusive)

    stop: int
        Stopping slice index (exclusive)

    stride: int
        Stride amount along that dimension

    Returns
    -------
    Tensor
        Forge tensor
    """
    if dim < 0:
        dim += len(operandA.shape)

    if stop is None:
        stop = start + 1

    if stop < 0:
        stop += operandA.shape[dim]

    # Handle negative stride by converting it to a positive value (simple case)
    if stride < 0:
        # NOTE: This simplified conversion is only valid when the size of the dimension is 1.
        # For dimensions with size larger than 1, proper flipping logic needs to be implemented.
        assert operandA.shape[dim] == 1, (
            f"Negative stride conversion is only supported for dimensions of size 1. "
            f"Got size {operandA.shape[dim]} for dim {dim}"
        )

        # Convert the negative stride to its absolute value
        stride = abs(stride)

        # Handle special case for flip-like operations: (start = -1, stop = very large negative value)
        # This pattern typically represents a full reverse slice from end to beginning.
        # Convert it to a standard forward slice: start = 0, stop = operandA.shape[dim]
        if start == -1 and stop >= torch.iinfo(torch.int64).min:
            start = 0
            stop = operandA.shape[dim]

    assert stride > 0

    assert start < operandA.shape[dim]
    assert stop <= operandA.shape[dim]
    assert stride <= operandA.shape[dim]

    return op(
        "index", name, operandA, attrs=(dim, start, stop, stride), dim=dim, start=start, stop=stop, stride=stride
    ).get_tensor()


def AdvIndex(
    name: str,
    operandA: Tensor,
    operandB: Tensor,
    dim: int = 0,
) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A - reference tensor

    operandA: Tensor
        Input operand B - indices

    dim: int
        Dimension to fetch indices over

    Returns
    -------
    Tensor
        Forge tensor
    """
    if dim < 0:
        dim += len(operandA.shape)

    return op("adv_index", name, operandA, operandB, dim=dim).get_tensor()


def Select(
    name: str,
    operandA: Tensor,
    dim: int,
    index: Union[int, Tuple[int, int]],
    stride: int = 0,
) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to slice

    index: int
        int: Index to select from that dimension
        [start: int, length: int]: Index range to select from that dimension

    stride: int
        Stride amount along that dimension

    Returns
    -------
    Tensor
        Forge tensor
    """
    dims = len(operandA.shape)
    if dim < 0:
        dim += dims

    assert dim < 4

    if type(index) is int:
        index = (index, 1)

    if stride == 0:
        stride = operandA.shape[dim]

    start, length = index
    assert start < operandA.shape[dim], f"start = {start} should be < operandA.shape[{dim}] = {operandA.shape[dim]}"
    assert (start + length) <= operandA.shape[
        dim
    ], f"(start = {start} + length = {length}) should be <= operandA.shape[{dim}] = {operandA.shape[dim]}"
    assert (
        stride <= operandA.shape[dim]
    ), f"stride = {stride} should be <= operandA.shape[{dim}] = {operandA.shape[dim]}"
    assert (start + length) <= stride, f"(start = {start} + length = {length}) should be <= stride = {stride}"
    assert (start + length) > 0, f"(start = {start} + length = {length}) should be > 0"

    return op(
        "select",
        name,
        operandA,
        attrs=(dim, index[0], index[1], stride),
        **{
            "dim": dim,
            "begin": index[0],
            "length": index[1],
            "stride": stride,
        },
    ).get_tensor()


def Pad(
    name: str,
    operandA: Tensor,
    pad: Tuple[int, ...],
    mode: str = "constant",
    value: float = 0.0,
    channel_last: bool = False,
) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A to which padding will be applied.

    pad: Tuple[int, ...]
        A tuple of padding values. The tuple should correspond to padding values for the tensor, such as
        [left, right, top, bottom].

    mode: str, optional
        The padding mode. Default is "constant". Other modes can be supported depending on the
        implementation (e.g., "reflect", "replicate").

    value: float, optional
        The value to use for padding when the mode is "constant". Default is 0.

    channel_last: bool, optional
        Whether the channel dimension is the last dimension of the tensor. Default is False.


    Returns
    -------
    Tensor
        A tensor with the specified padding applied to the input tensor.
    """
    assert (
        len(pad) == 2 or len(pad) == 4
    ), "Expect (padding_left, padding_right) or (padding_left, padding_right, padding_top, padding_bottom)"
    assert mode in [
        "constant",
        "replicate",
        "reflect",
    ], "Currently pad op only supports constant/replicate/reflect mode"

    assert not (
        mode in ["reflect", "replicate"] and len(operandA.shape) < 2
    ), "Padding mode 'reflect' and 'replicate' require at least 2 dimensions"

    mode_index = {
        "constant": 0,
        "replicate": 1,
        "reflect": 2,
    }

    named_attrs = {
        "padding": list(pad),
        "mode": mode_index[mode],
        "value": value,
        "channel_last": channel_last,
    }
    attrs = named_attrs["padding"] + [named_attrs["mode"], named_attrs["value"], named_attrs["channel_last"]]
    return op(
        "pad",
        name,
        operandA,
        attrs=attrs,
        **named_attrs,
    ).get_tensor()


def ConstantPad(
    name: str,
    operandA: Tensor,
    padding: List[int],
    value: float = 0.0,
) -> Tensor:
    """
    TM - Direct TTIR constant padding operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A to which padding will be applied.

    padding: List[int]
        Padding values in TTIR format: [dim0_low, dim0_high, dim1_low, dim1_high, ...]
        Length must be 2 * rank of input tensor.

    value: float, optional
        The constant value to use for padding. Default is 0.0.

    Returns
    -------
    Tensor
        A tensor with the specified constant padding applied to the input tensor.
    """
    assert len(padding) == 2 * len(
        operandA.shape
    ), f"Padding length {len(padding)} must be 2 * rank {len(operandA.shape)}"

    named_attrs = {
        "padding": padding,
        "value": value,
    }

    return op(
        "constant_pad",
        name,
        operandA,
        attrs=[],
        **named_attrs,
    ).get_tensor()


def Broadcast(name: str, operandA: Tensor, dim: int, shape: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to broadcast

    shape: int
        Output length of dim

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op("broadcast", name, operandA, dim=dim, size=shape).get_tensor()


def Repeat(name: str, operandA: Tensor, repeats: List[int]) -> Tensor:
    """
    Repeats this tensor along the specified dimensions.

    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat(4, 2)
    tensor([[ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3]])

    NOTE:
    -----
    This Forge.Repeat is equivalent to torch.repeat, numpy.tile, tvm.tile, and ttnn.repeat

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    factors: List[int]
        Multiplier on respective dim

    Returns
    -------
    Tensor
        Forge tensor
    """
    return op("repeat", name, operandA, repeats=repeats).get_tensor()


def RepeatInterleave(name: str, operandA: Tensor, repeats: int, dim: int) -> Tensor:
    """
    Repeat elements of a tensor.

    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat_interleave(2)
    tensor([1, 1, 2, 2, 3, 3])

    NOTE:
    -----
    This Forge.RepeatInterleave is equivalent to torch.repeat_interleave, numpy.repeat, tvm.repeat, and ttnn.repeat_interleave

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    repeats: int
        The number of repetitions for each element.

    dim: int
        The dimension along which to repeat values.

    Returns
    -------
    Tensor
        Forge tensor
    """
    return op(
        "repeat_interleave",
        name,
        operandA,
        repeats=repeats,
        dim=dim,
    ).get_tensor()


def Unsqueeze(name: str, operandA: Tensor, dim: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to broadcast

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op("unsqueeze", name, operandA, attrs=(dim,), dim=dim).get_tensor()


def Squeeze(name: str, operandA: Tensor, dim: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to broadcast

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op("squeeze", name, operandA, attrs=(dim,), dim=dim).get_tensor()


def ForgePad(name: str, operandA: Tensor, paddings: Tuple[int, int], value: float) -> Tensor:
    """
    Pad operation that expands a given tensor with arbitrary number of tiles by any dimension.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    paddings: Tuple[int, int]
        Tuple of paddings for R and C dimensions

    value: float
        Value to pad with
    """
    return op(
        "forge_pad",
        name,
        operandA,
        attrs=(paddings[0], paddings[1], value),
        pad_r=paddings[0],
        pad_c=paddings[1],
        value=value,
    ).get_tensor()


def ForgeUnpad(
    name: str,
    operandA: Tensor,
    original_length: Tuple[int, ...],
    paddings: Tuple[int, int],
) -> Tensor:
    """
    Unpad operation that removes arbitrary number of tiles by any dimension.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    original_length: Tuple[int, ...]
        Original length of R and C dimensions before padding

    paddings: Tuple[int, int]
        Tuple of paddings for R and C dimensions
    """
    return op(
        "forge_unpad",
        name,
        operandA,
        attrs=(paddings[0], paddings[1], original_length[0], original_length[1]),
        pad_r=paddings[0],
        pad_c=paddings[1],
        original_length_r=original_length[0],
        original_length_c=original_length[1],
    ).get_tensor()
