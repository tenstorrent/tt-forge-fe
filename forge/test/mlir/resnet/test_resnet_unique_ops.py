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
            (1, 64, 112, 112),  # Input shape
            3,  # kernel size
            2,  # stride size
            (1, 1, 1, 1),  # padding
            False,  # ceil mode
            marks=pytest.mark.xfail(
                reason="Tensor mismatch. Tracking on: https://github.com/tenstorrent/tt-mlir/issues/1575"
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


@pytest.mark.xfail(
    reason="RuntimeError: TT_FATAL @ /tt-metal/src/tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp:474: new_volume == old_volume. Invalid arguments to reshape. Tracking on: https://github.com/tenstorrent/tt-mlir/issues/1574"
)
@pytest.mark.push
def test_avg_pool2d_resnet():
    class AvgPool2d(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(
                x, kernel_size=[7, 7], stride=[7, 7], padding=(0, 0), ceil_mode=False, count_include_pad=True
            )

    inputs = [torch.rand(1, 2048, 7, 7)]

    framework_model = AvgPool2d()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "outer_dim_x, outer_dim_y, inner_dim",
    [
        pytest.param(
            1,  # Outer dimension x
            1000,  # Outer dimension y
            2048,  # Inner dimension
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

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param(
            (64,),  # Input shape
            1,  # Dimension to unsqueeze
        ),
        pytest.param(
            (128,),
            1,
        ),
        pytest.param(
            (256,),
            1,
        ),
        pytest.param(
            (512,),
            1,
        ),
        pytest.param(
            (1024,),
            1,
        ),
        pytest.param(
            (2048,),
            1,
        ),
    ],
)
@pytest.mark.push
def test_unsqueeze_resnet(input_shape, dim):
    class Unsqueeze(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.unsqueeze(a, dim)

    inputs = [torch.rand(*input_shape)]

    framework_model = Unsqueeze()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
