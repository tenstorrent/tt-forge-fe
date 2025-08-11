# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ..tensor import Tensor
from .common import ForgeOp as op


def FillCache(
    name: str,
    cache: Tensor,
    input: Tensor,
    batch_offset: int = 0,
) -> Tensor:
    """
    FillCache op writes the input into the cache tensor starting at the specified update index.

    Parameters
    ----------
    name: str
        Unique op name.

    cache: Tensor
        4D cache tensor of shape [B, H, S_total, D]

    input: Tensor
        4D input tensor of shape [B, H, S_input, D]

    update_idx: int
        The starting position in dim=2 to begin writing.

    batch_offset: int
        Offset in the batch dimension.
    """
    return op(
        "fill_cache",
        name,
        cache,
        input,
        batch_offset=batch_offset,
    ).get_tensor()


def UpdateCache(
    name: str,
    cache: Tensor,
    input: Tensor,
    update_index: int,
    batch_offset: int = 0,
) -> Tensor:
    """
    UpdateCache writes a single token (S=1) slice into the cache tensor on specified index.

    Parameters
    ----------
    name: str
        Unique op name.

    cache: Tensor
        4D cache tensor of shape [B, H, S_total, D]

    input: Tensor
        4D input tensor of shape [B, H, 1, D]

    update_idx: int
        Position in dim=2 to write the input slice.

    batch_offset: int
        Offset in the batch dimension.
    """
    return op(
        "update_cache",
        name,
        cache,
        input,
        update_index,
        batch_offset=batch_offset,
    ).get_tensor()
