# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..tensor import Tensor
from .common import ForgeOp as op


def ReduceSum(name: str, operandA: Tensor, dim: int, keep_dim: bool = True) -> Tensor:
    """
    Reduce by summing along the given dimension

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

    Returns
    -------
    Tensor
        Forge tensor
    """

    assert (dim >= -4) and (dim <= 3)
    # if dim < 0:
    #     dim += 4

    return op("reduce_sum", name, operandA, dim_arg=[dim], keep_dim=keep_dim).get_tensor()


def ReduceAvg(name: str, operandA: Tensor, dim: int, keep_dim: bool = True) -> Tensor:
    """
    Reduce by averaging along the given dimension

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

    Returns
    -------
    Tensor
        Forge tensor
    """

    assert (dim >= -4) and (dim <= 3)
    # if dim < 0:
    #     dim += 4

    return op("reduce_avg", name, operandA, dim_arg=[dim], keep_dim=keep_dim).get_tensor()


def ReduceMax(name: str, operandA: Tensor, dim: int, keep_dim: bool = True) -> Tensor:
    """
    Reduce by taking maximum along the given dimension

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

    Returns
    -------
    Tensor
        Forge tensor
    """
    assert (dim >= -4) and (dim <= 3)

    return op("reduce_max", name, operandA, dim_arg=[dim], keep_dim=keep_dim).get_tensor()


def Argmax(name: str, operandA: Tensor, dim: int = None, keep_dim=False) -> Tensor:
    """
    Argmax

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        The dimension to reduce (if None, the output is the argmax of the whole tensor)

    keep_dim: bool
        If True, retains the dimension that is reduced, with size 1.
        If False (default), the dimension is removed from the output shape.

    Returns
    -------
    Tensor
        Forge tensor
    """

    kwargs = {"keep_dim": keep_dim}

    if dim is not None:
        if dim < 0:
            dim += len(operandA.shape)
        kwargs["dim_arg"] = [dim]

    return op("argmax", name, operandA, **kwargs).get_tensor()


def Argsort(name: str, operandA: Tensor, axis: int = -1, is_ascend: bool = True, dtype: str = "int32") -> Tensor:
    """
    Argsort

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    axis: int
        The axis along which to sort

    is_ascend: bool
        Whether to sort in ascending order (True) or descending order (False)

    dtype: str
        The dtype of the output indices

    Returns
    -------
    Tensor
        Forge tensor
    """

    if axis < 0:
        axis += len(operandA.shape)

    return op("argsort", name, operandA, axis=axis, is_ascend=is_ascend, dtype=dtype).get_tensor()


def Sort(name: str, operandA: Tensor, axis: int = -1, is_ascend: bool = True) -> Tensor:
    """
    Sort

    Parameters
    ----------
    name: str
            Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
            Tensor to sort

    axis: int
            The axis along which to sort

    is_ascend: bool
            Whether to sort in ascending order (True) or descending order (False)

    Returns
    -------
    Tensor
            Forge tensor with values sorted along the specified axis
    """

    if axis < 0:
        axis += len(operandA.shape)

    return op("sort", name, operandA, axis=axis, is_ascend=is_ascend).get_tensor()
