# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ..common import to_torch_operands
from ..interface import PyOp


class UpdateCache(PyOp):
    @classmethod
    def create(cls, batch_offset=0):
        self = cls("update_cache")
        self.batch_offset = batch_offset
        return self

    def eval(self, tensors):
        cache, input, update_index = to_torch_operands(*tensors)

        if update_index.ndim == 0:
            update_index = update_index.unsqueeze(0)

        assert cache.ndim == 4, "Expected 4D tensor for cache"
        assert input.ndim == 4, "Expected 4D tensor for input"
        assert update_index.ndim == 1, "Expected 1D tensor for update_index"

        B_cache, H_cache, S_cache, D_cache = cache.shape
        B_in, H_in, S_in, D_in = input.shape
        # B is batch size, H is number of heads, S is sequence length, D is hidden dimension
        assert (
            S_in == 1
        ), "UpdateCache operation requires input sequence length S=1, but received a different length. Cache update can update only one position at a time."
        assert (H_cache, D_cache) == (
            H_in,
            D_in,
        ), "Number of heads H and hidden dimension D have to match for cache and input for UpdateCache op"

        assert (
            update_index.shape[0] == B_in
        ), f"Update index batch size {update_index.shape[0]} must match input batch size {B_in}"

        batch_offset = self.batch_offset
        assert (
            batch_offset + B_in <= B_cache
        ), f"batch_offset ({batch_offset}) + input batch size ({B_in}) exceeds cache batch size ({B_cache})"

        for b in range(B_in):
            idx = update_index[b].item()
            assert 0 <= idx < S_cache, f"Invalid update index {idx} at batch {b}"
            cache[b + batch_offset, :, idx : idx + S_in, :] = input[b]

        return cache

    def shape(self, tensor_shapes):
        cache_shape, _, _ = tensor_shapes
        return cache_shape, []

    def is_tm(self) -> bool:
        return False

    def is_eltwise(self) -> bool:
        return False

    def is_eltwise_binary(self) -> bool:
        return False

    def is_eltwise_unary(self) -> bool:
        return False

    def is_eltwise_nary(self) -> bool:
        return False


class FillCache(PyOp):
    @classmethod
    def create(cls, update_idx, batch_offset=0):
        self = cls("fill_cache")
        self.update_idx = update_idx
        self.batch_offset = batch_offset
        return self

    def eval(self, tensors):
        cache, input = to_torch_operands(*tensors)

        assert cache.ndim == 4, "Expected 4D tensor for cache"
        assert input.ndim == 4, "Expected 4D tensor for input"

        B_cache, H_cache, S_cache, D_cache = cache.shape
        B_in, H_in, S_in, D_in = input.shape

        # Ensure input shape aligns with cache shape
        assert (H_in, D_in) == (
            H_cache,
            D_cache,
        ), "Number of heads H and hidden dimension D must match for cache and input in FillCache"

        batch_offset = self.batch_offset
        assert (
            batch_offset + B_in <= B_cache
        ), f"batch_offset ({batch_offset}) + input batch size ({B_in}) exceeds cache batch size ({B_cache})"

        for b in range(B_in):
            assert S_in <= S_cache, f"Fill would write past the end of cache: S_in {S_in} > S_cache {S_cache}"
            cache[b + batch_offset, :, 0:S_in, :] = input[b]

        return cache

    def shape(self, tensor_shapes):
        (
            cache_shape,
            _,
        ) = tensor_shapes
        return cache_shape, []

    def is_tm(self) -> bool:
        return False

    def is_eltwise(self) -> bool:
        return False

    def is_eltwise_binary(self) -> bool:
        return False

    def is_eltwise_unary(self) -> bool:
        return False

    def is_eltwise_nary(self) -> bool:
        return False
