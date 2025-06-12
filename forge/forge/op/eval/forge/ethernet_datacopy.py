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


class EthernetDatacopy(PyEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("ethernet_datacopy")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "ethernet_datacopy should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = tensors[0]

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "ethernet_datacopy should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert False, f"ethernet_datacopy not defined in eltwise unary backward."

    def lower(self, lc, tensors, outputs):
        # TODO: Implement mlir lowering here.
        assert False
