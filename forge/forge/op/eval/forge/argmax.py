# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn.functional
from ..interface import PyEltwiseUnaryOp
from loguru import logger
from ..common import to_torch_operands
from ....forgeglobal import TILE_DIM
from ....tensor import forge_dataformat_to_pytorch_dtype
import numpy as np
from forge.op.eval.common import calculate_tile_size
from ..lforge.abs import Abs as ForgeAbs


class Argmax(PyEltwiseUnaryOp):
    @classmethod
    def create(cls, dim=None, keep_dim=False):
        self = cls("argmax")
        self.dim = dim
        self.keep_dim = keep_dim

        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Argmax should have one input"

        ret = torch.argmax(tensors[0], dim=self.dim, keepdim=self.keep_dim)

        return ret.to(dtype=torch.int32)

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Argmax should have one input"

        input_shape = tensor_shapes[0]

        # Dimension-specific argmax
        if self.dim is not None:
            if self.keep_dim:
                shape = list(input_shape)
                shape[self.dim] = 1
            else:
                shape = [d for i, d in enumerate(input_shape) if i != self.dim]
        else:  # Global argmax across all dimensions
            if self.keep_dim:
                shape = [1] * len(input_shape)  # All dimensions become 1
            else:
                shape = []  # Empty tuple for scalar result

        return tuple(shape), []

    def backward(self, ac, operand, inputs, output, grad):
        raise RuntimeError("Argmax does not require grad and does not have a backwards function")

    def lower(self, lc, tensors, outputs):
        return None

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops
