# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import random
import pytest

import forge.op
from forge import ForgeModule
from forge import Tensor, compile
from forge.verify.verify import verify


class UpdateCacheWrapper(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cache, input, update_index):
        return forge.op.UpdateCache("", cache, input, update_index)


class FillCacheWrapper(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cache, input):
        return forge.op.FillCache("", cache, input)


@pytest.mark.parametrize(
    "shapes_and_types",
    [
        # cache: [B=1, H=2, S=4, D=3]
        # input: [B=1, H=2, S=1, D=3]
        # update_index: [1]
        (
            ((1, 2, 4, 3), torch.float32),  # cache
            ((1, 2, 1, 3), torch.float32),  # input
        ),
    ],
)
@pytest.mark.push
def test_update_cache(shapes_and_types):
    cache_shape, cache_dtype = shapes_and_types[0]
    input_shape, input_dtype = shapes_and_types[1]

    max_update_index = cache_shape[2] - input_shape[2]
    update_idx_value = random.randint(0, max_update_index)

    cache_tensor = torch.zeros(cache_shape, dtype=cache_dtype)
    input_tensor = torch.ones(input_shape, dtype=input_dtype)
    update_index_tensor = torch.tensor([update_idx_value], dtype=torch.int32)

    cache = Tensor.create_from_torch(cache_tensor)
    input = Tensor.create_from_torch(input_tensor)
    update_index = Tensor.create_from_torch(update_index_tensor)

    inputs = [cache, input, update_index]

    model = UpdateCacheWrapper("update_cache_op")
    compiled = compile(model, sample_inputs=inputs)

    verify(
        inputs,
        model,
        compiled,
    )


@pytest.mark.parametrize(
    "shapes_and_types",
    [
        # cache: [B=1, H=2, S=6, D=3]
        # input: [B=1, H=2, S=3, D=3]
        # update_index: int (start pos in dim=2 to write input)
        (
            ((1, 2, 6, 3), torch.float32),  # cache
            ((1, 2, 3, 3), torch.float32),  # input
        ),
    ],
)
@pytest.mark.push
def test_fill_cache(shapes_and_types):
    cache_shape, cache_dtype = shapes_and_types[0]
    input_shape, input_dtype = shapes_and_types[1]

    cache_tensor = torch.zeros(cache_shape, dtype=cache_dtype)
    input_tensor = torch.ones(input_shape, dtype=input_dtype)

    cache = Tensor.create_from_torch(cache_tensor)
    input = Tensor.create_from_torch(input_tensor)

    inputs = [
        cache,
        input,
    ]

    model = FillCacheWrapper("fill_cache_op")
    compiled = compile(model, sample_inputs=inputs)

    verify(
        inputs,
        model,
        compiled,
    )
