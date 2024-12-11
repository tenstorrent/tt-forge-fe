# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from torch import nn

import forge
from forge.tensor import to_forge_tensors
from tvm.relay.op.transform import squeeze
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride_size, padding, ceil_mode",
    [
        pytest.param(
            (1, 64, 112, 112),
            3,
            2,
            (1, 1, 1, 1),
            False,
            marks=pytest.mark.xfail(
                reason="Tensor mismatch. PCC = 0.0025625005406270194, but required = 0.99. Tracking on "
            ),
        ),
    ],
)
@pytest.mark.push
def test_maxpool2d_resnet(input_shape, kernel_size, stride_size, padding, ceil_mode):
    class maxpool2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.padding = padding
            self.maxpool2d = nn.MaxPool2d(
                kernel_size=kernel_size, stride=stride_size, padding=0, dilation=1, ceil_mode=ceil_mode
            )

        def forward(self, x):
            if padding != 0:
                x = nn.functional.pad(x, self.padding, mode="constant", value=0)
            return self.maxpool2d(x)

    inputs = [torch.rand(input_shape).to(dtype=torch.bfloat16)]

    framework_model = maxpool2d().to(dtype=torch.bfloat16)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride_size, padding, ceil_mode",
    [
        pytest.param(
            (1, 64, 112, 112),
            3,
            2,
            (1, 1, 1, 1),
            False,
            marks=pytest.mark.xfail(
                reason="Tensor mismatch. PCC = 0.0025625005406270194, but required = 0.99. Tracking on "
            ),
        ),
    ],
)
@pytest.mark.push
def test_maxpool2d_resnet(input_shape, kernel_size, stride_size, padding, ceil_mode):
    class maxpool2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.padding = padding
            self.maxpool2d = nn.MaxPool2d(
                kernel_size=kernel_size, stride=stride_size, padding=0, dilation=1, ceil_mode=ceil_mode
            )

        def forward(self, x):
            if padding != 0:
                x = nn.functional.pad(x, self.padding, mode="constant", value=0)
            return self.maxpool2d(x)

    inputs = [torch.rand(input_shape).to(dtype=torch.bfloat16)]

    framework_model = maxpool2d().to(dtype=torch.bfloat16)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "outer_dim_x, outer_dim_y, inner_dim",
    [
        pytest.param(
            1,  # Outer dimension x
            1000,  # Outer dimension y
            2048,  # Inner dimension
            marks=pytest.mark.xfail(
                reason="Tensor mismatch. PCC = 0.9425581505871167, but required = 0.99. Tracking on: "
            ),
        ),
    ],
)
@pytest.mark.push
def test_matmul_resnet(outer_dim_x, outer_dim_y, inner_dim):
    class Matmul(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    inputs = [
        torch.rand(outer_dim_x, inner_dim),
        torch.rand(inner_dim, outer_dim_y),
    ]

    framework_model = Matmul()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_allclose=False))