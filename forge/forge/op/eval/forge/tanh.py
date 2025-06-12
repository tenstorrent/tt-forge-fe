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


class Tanh(PyEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("tanh")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Tanh should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = torch.tanh(tensors[0])

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Tanh should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        tanh_square = ac.op("multiply", (output, output))
        subtract = ac.op("subtract", (ac.constant(1), tanh_square))
        res = ac.op("multiply", (subtract, grad))
        return res

    def lower(self, lc, tensors, outputs):
        # TODO: Implement mlir lowering here.
        assert False

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops
