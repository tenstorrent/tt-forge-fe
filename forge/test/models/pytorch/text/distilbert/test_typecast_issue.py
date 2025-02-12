# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.nn import Module

import forge
from forge.verify.verify import verify


class MaskedAttention(Module):
    def __init__(self):
        super(MaskedAttention, self).__init__()

    def forward(self, scores, mask):
        mask_reshp = mask.unsqueeze(1)
        mask = (mask == 0).view(mask_reshp.shape).expand_as(scores)
        scores = scores.masked_fill(mask, torch.tensor(torch.finfo(scores.dtype).min, device=scores.device))
        return scores


@pytest.mark.parametrize(
    "batch_size, n_heads, q_length, k_length, mask_shape",
    [
        (2, 4, 5, 6, (2, 5, 6)),
        # (1, 8, 10, 12, (1, 10, 12)),
        # (3, 4, 7, 8, (3, 7, 8)),
        # (4, 2, 6, 10, (4, 6, 10)),
        # (2, 6, 8, 10, (2, 8, 10)),
    ],
)
def test_masked_attention(batch_size, n_heads, q_length, k_length, mask_shape):
    scores = torch.rand(batch_size, n_heads, q_length, k_length)
    mask = torch.randint(0, 2, mask_shape)

    framework_model = MaskedAttention()
    inputs = [scores, mask]

    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)
