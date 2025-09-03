# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Union

from forge._C.ops import OpType
from ..tensor import Tensor
from ..parameter import Parameter
from .common import ForgeOp as op


def Add(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise add of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Add)


def Subtract(name: str, operandA: Tensor, operandB: Tensor) -> Tensor:

    """
    Elementwise subtraction of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Subtract)


def Multiply(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:

    """
    Elementwise multiply of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Multiply)


def Divide(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:

    """
    Elementwise divide of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Divide)


def Max(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise max of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Maximum)


def Min(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise min of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Minimum)


def Heaviside(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise max of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Heaviside)


def Power(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    OperandA to the power of OperandB

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Power)


def Equal(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise equal of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Equal)


def NotEqual(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise equal of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.NotEqual)


def Greater(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise greater of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Greater)


def Less(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise less of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.Less)


def GreaterEqual(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise greater or equal of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.GreaterEqual)


def LessEqual(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise less or equal of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor

    """

    return _Eltwise(name, operandA, operandB, OpType.LessEqual)


def _Eltwise(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter], op_type: OpType) -> Tensor:

    """
    Common implementation for eltwise ops.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    op_type: OpType
        Operation type enum (OpType.Add, OpType.Subtract, OpType.Multiply...)

    Returns
    -------
    Tensor
        Forge tensor
    """

    result: Tensor = op(op_type, name, operandA, operandB).get_tensor()
    return result


def LogicalAnd(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:

    """
    Logical and operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return op(OpType.LogicalAnd, name, operandA, operandA).get_tensor()


def BitwiseAnd(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Bitwise and operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Forge tensor
    """

    return _Eltwise(name, operandA, operandB, OpType.BitwiseAnd)


def Remainder(name: str, operandA: Tensor, operandB: Union[Tensor, Parameter]) -> Tensor:
    return _Eltwise(name, operandA, operandB, OpType.Remainder)
