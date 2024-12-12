# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

from forge import compile
import pytest


@pytest.mark.parametrize(
    "shapes_dtypes",
    [
        (torch.tensor([1, 1, 1, 1, 32], dtype=torch.float32), torch.tensor([32], dtype=torch.float32)),
        (torch.tensor([32], dtype=torch.float32), torch.tensor([1, 1, 1, 1, 32], dtype=torch.float32)),
    ],
)
def test_add(shapes_dtypes):
    class AddOp(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_1, input_2):
            output = torch.add(input_1, input_2)
            return output

    inputs = shapes_dtypes

    framework_model = AddOp()
    framework_model.eval()

    compile(framework_model, sample_inputs=inputs)


def test_argmax():

    inputs = [torch.rand((1, 32, 64))]

    class ArgmaxOp(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_1):
            output = torch.argmax(input_1, dim=-1)
            return output

    framework_model = ArgmaxOp()
    framework_model.eval()

    compile(framework_model, sample_inputs=inputs)


def test_logsoftmax_torch():
    class LogSoftmaxOp(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_1):
            output = torch.nn.functional.log_softmax(input_1, dim=-1)
            return output

    inputs = [torch.rand((1, 32, 64))]

    framework_model = LogSoftmaxOp()
    framework_model.eval()

    compile(framework_model, sample_inputs=inputs)


def test_maximum():
    class MaximumOp(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_1, input_2):
            output = torch.maximum(input_1, input_2)
            return output

    inputs = [torch.randn((2, 3, 4)), torch.randn((2, 3, 4))]

    framework_model = MaximumOp()
    framework_model.eval()

    compile(framework_model, sample_inputs=inputs)


def test_avg_pool1d():
    class AvgPool1d(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.avg_pool1d(
                x, kernel_size=[7], stride=[7], padding=0, ceil_mode=False, count_include_pad=True
            )

    inputs = [torch.rand(1, 2048, 7)]

    framework_model = AvgPool1d()
    compile(framework_model, sample_inputs=inputs)
