# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from forge._C import DataFormat
from forge._C.ops import OpType
from ..tensor import Tensor
from .common import ForgeOp as op


def Constant(name: str, *, constant: float, dtype=DataFormat.Float32) -> Tensor:
    """
    Op representing user-defined constant

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    constant: float
        Constant value

    Returns
    -------
    Tensor
        Forge tensor
    """
    return op(OpType.Constant, name, **{"c": constant, "dtype": dtype}).get_tensor()
