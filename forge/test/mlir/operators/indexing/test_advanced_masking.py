# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "input_tensor, mask",
    [
        pytest.param(
            torch.arange(10, dtype=torch.float32),  # 1D tensor
            torch.tensor([True, False, True, False, True, False, True, False, True, False]),  # Mask
            id="1d_masked_select",
            marks=pytest.mark.xfail(reason="AssertionError: Dynamic shapes not supported"),
        ),
        pytest.param(
            torch.arange(16, dtype=torch.float32).reshape(4, 4),  # 2D tensor
            torch.tensor(
                [
                    [True, False, True, False],  # Mask
                    [False, True, False, True],
                    [True, True, False, False],
                    [False, False, True, True],
                ]
            ),
            id="2d_masked_select",
            marks=pytest.mark.xfail(reason="AssertionError: Dynamic shapes not supported"),
        ),
    ],
)
def test_masked_select(input_tensor, mask):
    class MaskedSelectModule(torch.nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.mask = mask

        def forward(self, x):
            # Apply masked_select
            return torch.masked_select(x, self.mask)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = MaskedSelectModule(mask)
    compiled_model = forge.compile(framework_model, inputs)

    # Verify outputs
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, mask, value",
    [
        pytest.param(
            torch.arange(10, dtype=torch.float32),  # 1D input tensor
            torch.tensor([True, False, True, False, False, True, False, True, False, False]),  # Mask
            -1.0,  # Value to fill
            id="1d_masked_fill",
        ),
        pytest.param(
            torch.arange(16, dtype=torch.float32).reshape(4, 4),  # 2D input tensor
            torch.tensor(
                [
                    [False, True, True, False],  # Mask
                    [True, False, False, True],
                    [False, False, True, True],
                    [True, True, False, False],
                ]
            ),
            99.0,  # Value to fill
            id="2d_masked_fill",
        ),
    ],
)
def test_masked_fill(input_tensor, mask, value):
    class MaskedFillModule(torch.nn.Module):
        def __init__(self, mask, value):
            super().__init__()
            self.mask = mask
            self.value = value

        def forward(self, x):
            # Apply masked_fill
            return x.masked_fill(self.mask, self.value)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = MaskedFillModule(mask, value)
    compiled_model = forge.compile(framework_model, inputs)

    # Verify outputs
    verify(inputs, framework_model, compiled_model)
