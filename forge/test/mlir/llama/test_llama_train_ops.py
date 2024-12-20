# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F
import pytest

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param(
            (1, 12, 200),
            -1,
        ),
        pytest.param(
            (128, 12, 200),
            -1,
        ),
        pytest.param(
            (256, 12, 200),
            -1,
        ),
        pytest.param(
            (1, 12, 200),
            -2,
        ),
        pytest.param(
            (128, 12, 200),
            -2,
        ),
        pytest.param(
            (256, 12, 200),
            -2,
        ),
    ],
)
@pytest.mark.push
@pytest.mark.xfail(
    reason="RuntimeError: Failed to run MLIR compiler pass pipeline. error: 'ttnn.reshape' op Shape attribute size must match output tensor rank. Tracking on: https://github.com/tenstorrent/tt-mlir/issues/1577"
)
def test_mean_bwd(input_shape, dim):
    class MeanBwd(nn.Module):
        def __init__(self, dim: int):
            super(MeanBwd, self).__init__()
            self.fc1 = nn.Linear(200, 3200)
            self.dim = dim

        def forward(self, x):
            return torch.mean(F.relu(self.fc1(x)), dim=self.dim)

    input_ids = torch.randn([*input_shape])

    framework_model = MeanBwd(dim=dim)
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=0.001)

    compiled_model = forge.compile(framework_model, input_ids, optimizer=framework_optimizer, training=True)

    verify([input_ids], framework_model, compiled_model)
