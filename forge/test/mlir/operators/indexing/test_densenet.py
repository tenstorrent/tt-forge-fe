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
            mask = x < self.threshold
            
            # Perform basic boolean indexing
            x[mask] = 7.0  # Simple assignment (modifies input tensor in-place)
            
            return x

    input_tensor[0, 1] = 0.0
    input_tensor[0, 2] = 0.0
    input_tensor[0, 3] = 0.0

    inputs = [input_tensor]
    framework_model = MinimalBooleanIndexModule()
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor",
    [
        pytest.param(
            torch.ones(1, 18, dtype=torch.float32),
            id="simplified_case",
        ),
    ],
)
def test_minimal_bool_indexing_no_change_of_input(input_tensor): # decomposes multiple ops one of which is argwhere
    class MaskingInputUnchanged(nn.Module):
        def __init__(self):
            super().__init__()
            self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        def forward(self, x):
            # Create simple mask
            mask = x < self.threshold
            
            # Perform basic boolean indexing

            y = x.clone() # we shouldn't change the input tensor in-place
            y[mask] = 7.0
            return y

    input_tensor[0, 1] = 0.0
    input_tensor[0, 2] = 0.0
    input_tensor[0, 3] = 0.0

    inputs = [input_tensor]
    framework_model = MaskingInputUnchanged()
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor",
    [
        pytest.param(
            torch.zeros(1, 3, dtype=torch.float32, requires_grad=False),
            id="simplified_case",
        ),
    ],
)
def test_oop_change(input_tensor): # decomposes multiple ops one of which is argwhere
    class OOPChange(nn.Module):
        def __init__(self):
            super().__init__()
            self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        def forward(self, x):
            y = x + 1
            x = x + 2

            return y
        

    inputs = [input_tensor]
    framework_model = OOPChange()
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)

@pytest.mark.parametrize(
    "input_tensor",
    [
        pytest.param(
            torch.zeros(1, 3, dtype=torch.float32, requires_grad=False),
            id="simplified_case",
        ),
    ],
)
def test_ip_change(input_tensor): # decomposes multiple ops one of which is argwhere
    class IPChange(nn.Module):
        def __init__(self):
            super().__init__()
            self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        def forward(self, x):
            y = x + 1
            x += 2

            z = y.sum()

            return z # problem with backward pass because of inplace operation
        

    inputs = [input_tensor]
    framework_model = IPChange()
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)
