# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

from forge._C.ops import OpType
from ..tensor import Tensor
from .common import ForgeOp as op


def CumSum(name: str, operandA: Tensor, dim: int) -> Tensor:

    """
    Cumulative sum operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    exclusive: bool
        Perform exclusive cumulative sum which includes (or not) the
        first operand. For example:
        x: [2, 4, 6, 8]

        cumsum(x, exclusive=False)
        [2, 6, 12, 20]

        cumsum(x, exclusive=True)
        [0,  2,  6, 12]

    Returns
    -------
    Tensor
        Forge tensor
    """
    if dim < 0:
        dim += len(operandA.shape)

    return op(OpType.CumulativeSum, name, operandA, dim=dim).get_tensor()
