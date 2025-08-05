# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from loguru import logger
import forge
from forge.utils import align_up_tile, align_up, round_up_div, clamp
from ...forgeglobal import TILE_DIM
from forge._C import DataFormat, compress_sparse_tensor_and_strip_info, SparseCOO, SparseFORGE, MathFidelity


def hstack(t: torch.Tensor, stack_factor: int):
    assert t.is_sparse, "this works only for sparse tensors"
    assert len(t.shape) == 4

    if not t.is_coalesced():
        t = t.coalesce()

    assert t.shape[1] % stack_factor == 0, "invalid hstack stack factor"
    slices_count = t.shape[1] // stack_factor
    ret = [[] for _ in range(5)]

    ws, zs, rows, cols = t.indices().tolist()
    vals = t.values().tolist()
    for idx in range(len(rows)):
        slice_idx = zs[idx] // stack_factor
        ret[0].append(ws[idx])
        ret[1].append(slice_idx)
        ret[2].append(rows[idx])
        ret[3].append(cols[idx] + (zs[idx] % stack_factor) * t.shape[-1])
        ret[4].append(vals[idx])

    ret = torch.sparse_coo_tensor(
        indices=[ret[0], ret[1], ret[2], ret[3]],
        values=ret[4],
        size=(t.shape[0], slices_count, t.shape[2], t.shape[3] * stack_factor),
        dtype=t.dtype,
    )
    return ret
