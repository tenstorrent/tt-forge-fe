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
            torch.ones(1, 18, dtype=torch.float32),
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
            # Create simple mask
            mask = x > self.threshold # there is a bug fwith x > self.threshold where it recognizes it as a less than operation :(
            
            # Perform basic boolean indexing
            x[mask] = 0.0  # Simple assignment instead of division
            
            return x

    input_tensor[0, 1] = 0.0
    input_tensor[0, 2] = 0.0
    input_tensor[0, 3] = 0.0

    inputs = [input_tensor]
    framework_model = MinimalBooleanIndexModule()
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)