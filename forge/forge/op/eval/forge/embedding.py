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


def lower(type, attr, lc, ops, outputs):
    assert type == "embedding"
    assert len(ops) == 2

    lc.set_output_df(ops[0], DataFormat.RawUInt32)
    lc.set_runtime_tensor_transform(ops[0], RuntimeTensorTransform.EmbeddingIndex(ops[0].shape))

    embedding_dim = ops[1].shape.as_list()
    while len(embedding_dim) < 4:
        embedding_dim = [1] + embedding_dim

    forge_attrs = {
        "num_indices": ops[0].shape[-1],
    }

    lc.op(type, ops, (ops[0].shape[-1],), forge_attrs, "", TILE_DIM, TILE_DIM)


def decompose(type, attr, dc, inputs):
    pass


def backward(type, attr, ac, operand, inputs, output, grad):
    assert type == "embedding"
    embedding_bw_context = ac.op("embedding_bw", [inputs[0], inputs[1], grad])
    ac.set_output_df(embedding_bw_context, inputs[1].output_df)
    return embedding_bw_context
