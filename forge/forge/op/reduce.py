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
