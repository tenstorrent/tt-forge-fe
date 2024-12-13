# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from ..common import to_torch_operands
from forge._C import DataFormat
from forge._C.graph import RuntimeTensorTransform, RuntimeTensorTransformType
from ....forgeglobal import TILE_DIM


def eval(type, attr, ops):
    assert type == "embedding_bw"
    assert len(ops) == 2
    t_ops = to_torch_operands(*ops)
    input = t_ops[0]
    weight = t_ops[1]
    grad = t_ops[1]

    result = torch.zeros(weight.shape)
    for i, idx in enumerate(input):
        result[idx] = grad[i]
    return result


def shape(type, attr, tensor_shapes):
    assert type == "embedding_bw"
    return tensor_shapes[1], []


def lower(type, attr, lc, ops, outputs):
    assert False, "embedding_bw should not be lowered"


def decompose(type, attr, dc, inputs):
    assert False, "embedding_bw should not be decomposed"


def backward(type, attr, ac, operand, inputs, output, grad):
    assert False, "embedding_bw should not be backwarded"
