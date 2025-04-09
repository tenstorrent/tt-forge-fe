# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig


@pytest.mark.parametrize("shape", [(1, 2, 3, 4)])
@pytest.mark.parametrize("operator_name", ["ge", "ne", "gt", "lt"])
@pytest.mark.parametrize("verify_dtype", [False, True])
def test_verify_dtype_logical_ops(forge_property_recorder, shape, operator_name, verify_dtype):

    class Model(nn.Module):
        def __init__(self, operator):
            super().__init__()
            self.operator = operator

        def forward(self, x, y):
            return self.operator(x, y)

    x = torch.rand(shape)
    y = torch.rand(shape)
    inputs = [x, y]

    operator = getattr(torch, operator_name)
    framework_model = Model(operator)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(verify_dtype=verify_dtype),
        forge_property_handler=forge_property_recorder,
    )
    