# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "input_tensor, param",
    [
        pytest.param(
            torch.tensor([10, 20, 30, 40], dtype=torch.float32),  # 1D tensor
            20.0,  # Threshold value
            id="1d_greater_than",
            marks=pytest.mark.xfail(reason='Failed on "DecomposeMultiDimSqueeze" TVM callback'),
        ),
        pytest.param(
            torch.arange(9, dtype=torch.float32).reshape(3, 3),  # 2D tensor
            4.0,  # Threshold value
            id="2d_greater_than",
            marks=pytest.mark.xfail(reason="AssertionError: Currently supportes only 1D tensors"),
        ),
    ],
)
@pytest.mark.push
@pytest.mark.functional
def test_masking_greater_than(forge_property_recorder, input_tensor, param):
    class GreaterThanMaskingModule(torch.nn.Module):
        def __init__(self, param):
            super().__init__()
            self.param = param

        def forward(self, x):
            return x[x > self.param]

    # Expected result
    expected_result = input_tensor[input_tensor > param]

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = GreaterThanMaskingModule(param)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_tensor, mod_value",
    [
        pytest.param(
            torch.arange(9, dtype=torch.float32),  # 1D tensor
            2,  # Modulus value
            id="1d_modulus",
            marks=pytest.mark.xfail(reason='Failed on "DecomposeMultiDimSqueeze" TVM callback'),
        ),
        pytest.param(
            torch.arange(9, dtype=torch.float32).reshape(3, 3),  # 2D tensor
            2,  # Modulus value
            id="2d_modulus",
            marks=pytest.mark.xfail(reason="AssertionError: Currently supportes only 1D tensors"),
        ),
    ],
)
@pytest.mark.push
@pytest.mark.functional
def test_masking_modulus(forge_property_recorder, input_tensor, mod_value):
    class ModulusMaskingModule(torch.nn.Module):
        def __init__(self, mod_value):
            super().__init__()
            self.mod_value = mod_value

        def forward(self, x):
            return x[x % self.mod_value == 0]

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = ModulusMaskingModule(mod_value)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_tensor, greater_param, mod_param",
    [
        pytest.param(
            torch.arange(12, dtype=torch.float32),  # 1D tensor
            5,  # Greater-than parameter
            2,  # Modulus parameter
            id="1d_combined",
            marks=pytest.mark.xfail(reason='Failed on "DecomposeMultiDimSqueeze" TVM callback'),
        ),
        pytest.param(
            torch.arange(16, dtype=torch.float32).reshape(4, 4),  # 2D tensor
            6,  # Greater-than parameter
            3,  # Modulus parameter
            id="2d_combined",
            marks=pytest.mark.xfail(reason="AssertionError: Currently supportes only 1D tensors"),
        ),
    ],
)
@pytest.mark.push
@pytest.mark.functional
def test_masking_combined_conditions(forge_property_recorder, input_tensor, greater_param, mod_param):
    class CombinedMaskingModule(torch.nn.Module):
        def __init__(self, greater_param, mod_param):
            super().__init__()
            self.greater_param = greater_param
            self.mod_param = mod_param

        def forward(self, x):
            # Apply both greater-than and modulus masks
            return x[(x > self.greater_param) & (x % self.mod_param == 0)]

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = CombinedMaskingModule(greater_param, mod_param)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
