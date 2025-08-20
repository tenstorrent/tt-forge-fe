# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple
from math import gcd
import torch
from ..common import to_torch_operands
from forge.forgeglobal import TILE_DIM, align_up_tile
from forge.forgeglobal import TILE_DIM, align_up_tile, is_tile_dim_aligned
from loguru import logger


def eval(type, attr, ops):

    if type == "index_copy":
        t_ops = to_torch_operands(*ops)
        t_ops = list(t_ops)
        # index_copy expects index to be long tensor, not int
        t_ops[1] = t_ops[1].to(torch.long)
        out = t_ops[0].index_copy(attr[0], t_ops[1], t_ops[2])
        return out


# Return shape, and list of dimensions that were broadcast on operands
def shape(type, attr, ops) -> Tuple[Tuple, List]:
    if type == "index_copy":
        # index copy writes data to specified indices in the first operand
        # so the output shape is the same as the first operand
        return ops[0], []

    assert False, f"{type} not defined in eltwise_nary"


def backward(op_type, attr, ac, operand, inputs, output, grad):
    assert False, f"{op_type} not defined in eltwise_nary"


def decompose(type, attr, dc, inputs):

    if type == "index_copy":
        assert len(inputs) == 3, "Index copy should have 3 inputs"
        operandA, index, value = inputs
        assert len(attr) == 1, "Index copy should have 1 attr"
        dim = attr[0]
        # change dim to negative indexing
        if dim > 0:
            dim -= len(operandA.shape)
        if dim == -2 and len(operandA.shape) == 4 and len(value.shape) == 4:
            # If index contains more than one element, we consider decomposing to FillCache
            if index.shape[0] > 1:
                logger.warning(
                    "If the index operand in index_copy contains values that are not contiguous starting from 0, decomposing to FillCache will result in incorrect behavior. This is because FillCache fills continuously starting from index 0."
                )
                # FillCache is used to fill operandA from the beginning
                result = dc.op_with_named_attrs("fill_cache", [operandA, value], {"batch_offset": 0})
            else:
                # Single index case -> decompose to UpdateCache
                result = dc.op_with_named_attrs("update_cache", [operandA, value, index], {"batch_offset": 0})
        else:
            # Only index_copy with dim -2, and tensors of shape 4D can be decomposed to FillCache or UpdateCache
            # Leave index_copy as is
            return
        dc.fuse(result)


from math import gcd
from functools import reduce


def find_gcd(list):
    x = reduce(gcd, list)
    return x


def decompose_post_optimize(type, attr, dc, inputs):
    pass
