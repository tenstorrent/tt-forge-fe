# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

import forge
from forge.verify.verify import verify


@pytest.mark.push
@pytest.mark.parametrize(
    "input_tensor",
    [
        torch.randn(1, 14, 35, 35),
        torch.randn(1, 14, 35, 35) * 50 + 10,
    ],
)
def test_softmax(input_tensor):
    class Softmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, a):
            return self.softmax(a)

    inputs = [input_tensor]

    framework_model = Softmax()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
