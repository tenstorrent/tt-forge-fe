# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify

@pytest.mark.parametrize(
    "input_tensor",
    [
        pytest.param(
            torch.randn(1, 18, dtype=torch.float32),
            id="simplified_case",
        ),
    ],
)
def test_minimal_bool_indexing(input_tensor): # decomposes multiple ops one of which is argwhere
    class MinimalBooleanIndexModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        def forward(self, x):
            # Create simple mask - just one comparison
            mask = x < self.threshold
            
            # Perform basic boolean indexing
            result = x.clone()
            result[mask] = 0.0  # Simple assignment instead of division
            
            return result

    inputs = [input_tensor]
    framework_model = MinimalBooleanIndexModule()
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)