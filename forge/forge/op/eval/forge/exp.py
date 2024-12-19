# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torch.nn.functional
from loguru import logger

from forge.op.eval.common import calculate_tile_size

from ....forgeglobal import TILE_DIM
from ....tensor import forge_dataformat_to_pytorch_dtype
from ..common import to_torch_operands
from ..interface import PyEltwiseUnaryOp
from ..lforge.exp import Exp as ForgeExp


class Exp(PyEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("exp")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Exp should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = torch.exp(tensors[0])

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Exp should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Exp should have one input"
        assert operand == 0, "Invalid operand index"
        return ac.op("multiply", (output, grad))

    def lower(self, lc, tensors, outputs):
        assert len(tensors) == 1, "Exp should  have one input"
        approximate_mode = "true" if "FORGE_EXP_APPROX" in os.environ else "false"
        lc.op(ForgeExp.create(approximate_mode=approximate_mode), tensors)

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops
