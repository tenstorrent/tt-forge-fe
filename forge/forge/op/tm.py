# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union, Tuple, List, Dict
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

    return op("reshape", name, operandA, attrs=shape, shape=shape).get_tensor()


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
    if dim >= 0:
        dim -= len(operandA.shape)

    if stop is None:
        stop = start + 1

    if stop < 0:
        stop += operandA.shape[dim]

    assert stride > 0

    assert start < operandA.shape[dim]
    assert stop <= operandA.shape[dim]
    assert stride <= operandA.shape[dim]

    return op("index", name, operandA, attrs=(dim, start, stop, stride)).get_tensor()


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
    return op("adv_index", name, operandA, operandB, attrs=(dim,)).get_tensor()


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
    pad_len: int,
    mode: str = "constant",
    channel_last: bool = False,
    value: int = 0,
) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    pad: tuple
        Either (padding_left, padding_right) or (padding_left, padding_right, padding_top, padding_bottom))

    Returns
    -------
    Tensor
        Forge tensor
    """
    assert (
        len(pad) == 2 or len(pad) == 4
    ), "Expect (padding_left, padding_right) or (padding_left, padding_right, padding_top, padding_bottom)"
    assert mode in [
        "constant",
        "replicate",
        "reflect",
    ], "Currently pad op only supports constant/replicate/reflect mode"

    # breakpoint()
    rank = len(operandA.shape)
    pad_len = len(pad)
    if pad_len ==2:
        forge_pad = [0]*(2*rank)
        # breakpoint()
        forge_pad[-pad_len:] = pad
    elif pad_len == 4:
        forge_pad = [0]*(2*rank)
        forge_pad[-pad_len:] = pad
    #(left,right,top,bottom)
    # top,bottom -> height
    # left,right -> width

    # left,right -> (left,right)
    # breakpoint()
    return op(
        "pad",
        name,
        operandA,
        padding=forge_pad,
        mode=mode,
        channel_last=channel_last,
        value = float(value),
        pad_len = pad_len
    ).get_tensor()
    # 1x96x56x56
    # 1,96,54,54
    # 
# (0,0,0,0,0,0,0,0)
# 025-03-12 17:15:43.963 | INFO     | MLIRCompiler    - Generated MLIR for node pad_0 with value %1 = "ttir.pad"(%arg0, %0) <{padding = array<i32: 0, 0, 0, 0, 1, 1, 0, 0>, value = 0.000000e+00 : f32}> {channel_last = false, mode = "constant", pad_len = 4 : si32} : (tensor<1x96x54x54xf32>, tensor<1x96x54x56xf32>) -> tensor<1x96x54x56xf32>
# 2025-03-12 17:15:43.963 | INFO     | MLIRCompiler    - Skipping node Pad0.output_pad_0 as it is not a TTForge operation.
# loc("pad_0"("forward":4294967295:5)): error: 'ttir.pad' op Output tensor shape (1,96,54,56) must match the inferred shape: (1,96,56,54)

# 2025-03-12 17:24:47.251 | INFO     | MLIRCompiler    - Generated MLIR for node pad_0 with value %1 = "ttir.pad"(%arg0, %0) <{padding = array<i32: 0, 0, 0, 0, 0, 0, 1, 1>, value = 0.000000e+00 : f32}> {channel_last = false, mode = "constant", pad_len = 4 : si32} : (tensor<1x96x54x54xf32>, tensor<1x96x56x54xf32>) -> tensor<1x96x56x54xf32>
# 2025-03-12 17:24:47.251 | INFO     | MLIRCompiler    - Skipping node Pad0.output_pad_0 as it is not a TTForge operation.
# loc("pad_0"("forward":4294967295:5)): error: 'ttir.pad' op Output tensor shape (1,96,56,54) must match the inferred shape: (1,96,54,56)
# loc("Pad0":0:0): error: module verification failed.

# # (_,_,_,_,Top,Bottom,left,right)
# def Pad(
#     name: str,
#     operandA: Tensor,
#     pad: Union[Tuple[int, int, int, int], Tuple[int, int]],
#     mode: str = "constant",
#     channel_last: bool = False,
# ) -> Tensor:
#     """
#     TM

#     Parameters
#     ----------
#     name: str
#         Op name, unique to the module, or leave blank to autoset

#     operandA: Tensor
#         Input operand A

#     pad: tuple
#         Either (padding_left, padding_right) or (padding_left, padding_right, padding_top, padding_bottom))

#     Returns
#     -------
#     Tensor
#         Forge tensor
#     """
#     assert (
#         len(pad) == 2 or len(pad) == 4
#     ), "Expect (padding_left, padding_right) or (padding_left, padding_right, padding_top, padding_bottom)"
#     assert mode in [
#         "constant",
#         "replicate",
#         "reflect",
#     ], "Currently pad op only supports constant/replicate/reflect mode"

#     mode_index = {
#         "constant": 0,
#         "replicate": 1,
#         "reflect": 2,
#     }

#     attrs = list(pad) + [mode_index[mode], channel_last]
#     return op(
#         "pad",
#         name,
#         operandA,
#         attrs=attrs,
#     ).get_tensor()


def PadTile(name: str, operandA: Tensor, dim: int, original_length: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension which to pad to tile dim

    original_length: int
        Original length of the dimension before calling this function

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op("pad_tile", name, operandA, attrs=(dim, original_length)).get_tensor()


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

    return op("broadcast", name, operandA, attrs=(dim, shape, True)).get_tensor()


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
    return op("repeat", name, operandA, attrs=repeats, repeats=repeats).get_tensor()


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
        attrs=(repeats, dim),
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

    return op("unsqueeze", name, operandA, attrs=(dim, len(operandA.shape)), dim=dim).get_tensor()


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


def Narrow(name: str, operandA: Tensor, dim: int, start: int, length: int, original_length: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension which to pad to tile dim

    start: int
        Start index in the dimension to be narrowed

    length: int
        Number of items to take from start

    original_length: int
        Original length of the dimension before calling this function

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op("narrow", name, operandA, attrs=(dim, start, length, original_length)).get_tensor()


def PixelShuffle(name: str, operandA: Tensor, upscale_factor: int) -> Tensor:
    """
    Pixel shuffle operation.

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
    return op("pixel_shuffle", name, operandA, attrs=(upscale_factor,)).get_tensor()


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
    return op("forge_pad", name, operandA, attrs=(paddings[0], paddings[1], value)).get_tensor()


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
    ).get_tensor()
