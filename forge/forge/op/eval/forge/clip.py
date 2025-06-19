# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn.functional
from ..interface import PyEltwiseUnaryOp
from loguru import logger
from ..common import to_torch_operands
from ....forgeglobal import TILE_DIM
from ....tensor import forge_dataformat_to_pytorch_dtype
import numpy as np
from forge.op.eval.common import calculate_tile_size


class Clip(PyEltwiseUnaryOp):
    @classmethod
    def create(cls, min=float("-inf"), max=float("inf")):
        self = cls("clip")
        self.min = min
        self.max = max

        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Clip should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = torch.clip(tensors[0], min=self.min, max=self.max)

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Clip should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Clip should have one input"
        assert operand == 0, "Invalid operand index"
        heaviside = ac.op("heaviside", (inputs[0], ac.constant(0.5)))
        subtract = ac.op("subtract", (heaviside, ac.constant(0.5)))
        stretched = ac.op("multiply", (subtract, ac.constant(2.0)))
        return ac.op("multiply", (stretched, grad))

    def backward(self, ac, operand, inputs, output, grad):
        x = inputs[0]
        shape = x.shape.as_list()
        min_value = self.min
        max_value = self.max
        min_value_tensor = ac.tensor(torch.zeros(shape) + min_value)
        max_value_tensor = ac.tensor(torch.zeros(shape) + max_value)

        ge_x = ac.op("greater_equal", (x, min_value_tensor))
        le_x = ac.op("less_equal", (x, max_value_tensor))
        mask = ac.op("multiply", (ge_x, le_x))
        res = ac.op("multiply", (mask, grad))
        return res
