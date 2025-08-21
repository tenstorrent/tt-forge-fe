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
    pass


# Return shape, and list of dimensions that were broadcast on operands
def shape(type, attr, ops) -> Tuple[Tuple, List]:
    assert False, f"{type} not defined in eltwise_nary"


def backward(op_type, attr, ac, operand, inputs, output, grad):
    assert False, f"{op_type} not defined in eltwise_nary"


def decompose(type, attr, dc, inputs):
    pass


from math import gcd
from functools import reduce


def find_gcd(list):
    x = reduce(gcd, list)
    return x


def decompose_post_optimize(type, attr, dc, inputs):
    pass
