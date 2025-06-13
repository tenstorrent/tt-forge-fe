# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from ..common import to_torch_operands
from forge._C import DataFormat
from forge._C.graph import RuntimeTensorTransform, RuntimeTensorTransformType
from ....forgeglobal import TILE_DIM


def eval(type, attr, ops):
    assert type == "embedding"
    assert len(ops) == 2
    t_ops = to_torch_operands(*ops)
    return torch.embedding(t_ops[1], t_ops[0].to(torch.int32))


def shape(type, attr, ops):
    assert type == "embedding"
    assert len(ops) == 2
    shape = list(ops[0])
    shape.append(ops[1][-1])
    return shape, []


def lower(type, attr, ops, outputs):
    # TODO: Implement mlir lowering here.
    assert False


def decompose(type, attr, dc, inputs):
    pass


def backward(type, attr, ac, operand, inputs, output, grad):
    assert type == "embedding"
    embedding_bw_context = ac.op("embedding_bw", [inputs[0], inputs[1], grad])
    # Output data format should match second operand data format (embedding weights)
    ac.set_output_df(embedding_bw_context, inputs[1].output_df)
    return embedding_bw_context
