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


class Cosine(PyEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("cosine")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Cosine should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]

        ret = torch.cos(tensors[0])

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Cosine should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert False, f"Cosine not defined in eltwise unary backward."

    def lower(self, tensors, outputs):
        # TODO: Implement mlir lowering here.
        assert False

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops
