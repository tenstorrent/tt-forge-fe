# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from forge._C.ops import OpType
from ..tensor import Tensor
from ..parameter import Parameter
from .common import ForgeOp as op


def Concatenate(name: str, *operands: Tensor, axis: int) -> Tensor:

    """
    Concatenate tensors along axis

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operands: Tuple[Tensor, ...]
        tensors to be concatenated

    axis: int
        concatenate axis


    Returns
    -------
    Tensor
        Forge tensor
    """

    result: Tensor = op(OpType.Concatenate, name, *operands, dim=axis).get_tensor()
    return result


def Where(name: str, condition: Tensor, x: Tensor, y: Tensor) -> Tensor:

    """
    Returns elements selected from either x or y depending on condition

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    condition: Tensor
        When True (nonzero), yield x, else y

    x: Tensor
        value(s) if true

    y: Tensor
        value(s) if false

    Returns
    -------
    Tensor
        Forge tensor
    """

    result: Tensor = op(OpType.Where, name, condition, x, y).get_tensor()
    return result


def IndexCopy(name: str, operandA: Tensor, index: Tensor, value: Tensor, dim: int) -> Tensor:
    """
    Copies the elements of value into operandA at index along dim

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    index: Tensor
        Index at which to write into operandA

    value: Tensor
        Value to write out

    dim: int
        Dimension to broadcast

    Returns
    -------
    Tensor
        Forge tensor
    """
    if dim < 0:
        dim += len(operandA.shape)
    return op(OpType.IndexCopy, name, operandA, index, value, dim=dim).get_tensor()


def Stack(name: str, *operands: Tensor, axis: int) -> Tensor:

    """
    Stack tensors along new axis

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operands: Tuple[Tensor, ...]
        tensors to be stacked

    axis: int
        new stack axis


    Returns
    -------
    Tensor
        Forge tensor
    """

    result: Tensor = op(OpType.Stack, name, *operands, dim=axis).get_tensor()
    return result
