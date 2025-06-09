# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..tensor import Tensor
from .common import ForgeOp as op
from typing import Union, Tuple, List
from eval.interface import PyEltwiseUnaryOp
from ..parameter import Parameter
import forge
from forge._C import DataFormat
from forge._C.graph import OpType

class Constant(PyEltwiseUnaryOp):
    def __init__(
        self, op_type: str, *operands: Union[Tensor, Parameter], attrs: Tuple[int, ...] = (), **named_attrs
    ) -> Tensor:
        """
        Create an op with given parameters.
        """
        operands = tuple(
            forge.op.Constant("", constant=operand) if isinstance(operand, (int, float)) else operand
            for operand in operands
        )
        self.op_type = op_type
        self.operands = operands
        self.attrs = attrs
        self.named_attrs = named_attrs
        self.cpp_op_type = OpType(self.op_type, self.attrs, self.named_attrs)
    # Waiting on Vanja to merge simplifying of get_tensor

    def shape(type, attr, ops, tile_height, tile_width):
        assert len(ops) == 0, "constant should not have any operands"
        assert len(attr) == 1, "constant should contain single attr repr the const. val"
        return SINGLE_TILE_SHAPE, []


    def eval(type, attr, ops):
        assert len(ops) == 0, "constant should not have any operands"
        assert len(attr) == 1, "constant should contain single attr repr the const. val"

        # TODO: add data format
        const_tensor = torch.zeros(SINGLE_TILE_SHAPE)
        const_tensor[0, 0, 0, 0] = attr[0]

        return const_tensor

def Constant(name: str, *, constant: float) -> Tensor:
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
    return op("constant", name, attrs=(constant,)).get_tensor()