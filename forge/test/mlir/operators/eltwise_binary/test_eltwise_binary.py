# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - Atan"
)
@pytest.mark.parametrize(
    "shape",
    [
        (300, 1),
        (1, 6, 18),
        (2, 2, 2),
        (5, 5),
        (745),
        (1, 256, 6, 6),
        (1, 512, 14, 14),
        (1, 3, 224, 224),
        (1, 34, 200, 224, 53),
    ],
)
@pytest.mark.push
def test_atan2(shape):
    class Atan2(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return torch.atan2(x2, x1)

    inputs = [torch.randn(shape), torch.randn(shape)]
    framework_model = Atan2()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape_x, shape_y",
    [
        ((1, 128, 28, 28), (1, 128, 28, 28)),
        ((1, 64, 28, 28), (1, 64, 28, 28)),
        ((1, 256, 28, 28), (1, 256, 28, 28)),
        ((1, 128, 14, 14), (1, 128, 14, 14)),
        ((1, 128, 56, 56), (1, 128, 56, 56)),
        ((1, 32, 64, 64), (1, 32, 64, 64)),
        ((1, 512, 7, 7), (1, 512, 7, 7)),
        ((1, 32, 32, 32), (1, 32, 32, 32)),
    ],
)
@pytest.mark.push
def test_less(shape_x, shape_y):
    class Less(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.less(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = Less()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False))


@pytest.mark.parametrize(
    "shape_x, shape_y",
    [
        ((1, 128, 28, 28), (1, 128, 28, 28)),
        ((1, 64, 28, 28), (1, 64, 28, 28)),
        ((1, 256, 28, 28), (1, 256, 28, 28)),
        ((1, 128, 14, 14), (1, 128, 14, 14)),
        ((1, 128, 56, 56), (1, 128, 56, 56)),
        ((1, 32, 64, 64), (1, 32, 64, 64)),
        ((1, 512, 7, 7), (1, 512, 7, 7)),
        ((1, 32, 32, 32), (1, 32, 32, 32)),
    ],
)
@pytest.mark.push
def test_greater(shape_x, shape_y):
    class Greater(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.greater(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = Greater()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False))


@pytest.mark.parametrize(
    "shape_x, shape_y",
    [
        ((1, 128, 28, 28), (1, 128, 28, 28)),
        ((1, 64, 28, 28), (1, 64, 28, 28)),
        ((1, 256, 28, 28), (1, 256, 28, 28)),
        ((1, 128, 14, 14), (1, 128, 14, 14)),
        ((1, 128, 56, 56), (1, 128, 56, 56)),
        ((1, 32, 64, 64), (1, 32, 64, 64)),
        ((1, 512, 7, 7), (1, 512, 7, 7)),
        ((1, 32, 32, 32), (1, 32, 32, 32)),
    ],
)
@pytest.mark.push
def test_not_equal(shape_x, shape_y):
    class NotEqual(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.ne(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = NotEqual()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 28, 28),
        (1, 64, 28, 28),
        (1, 256, 28, 28),
        (1, 128, 14, 14),
        (1, 128, 56, 56),
        (1, 32, 64, 64),
        (1, 512, 7, 7),
        (1, 32, 32, 32),
        (128, 28, 28),
        (64, 28, 28),
        (256, 28, 28),
        (128, 14, 14),
        (128, 56, 56),
        (32, 64, 64),
        (512, 7, 7),
        (32, 32, 32),
        (128, 28),
        (64, 28),
        (256, 28),
        (128, 14),
        (128, 56),
        (32, 64),
        (512, 7),
        (32, 32),
    ],
)
@pytest.mark.push
def test_equal(shape):
    class Equal(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.eq(x, y)

    x = torch.rand(shape)
    y = x * 2.0

    inputs = [x, y]

    framework_model = Equal()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False))


@pytest.mark.push
def test_add():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    inputs = [torch.rand(2, 32, 32, dtype=torch.bfloat16), torch.rand(2, 32, 32, dtype=torch.float32)]

    framework_model = Add()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("dims", [(1, 32, 64), (6, 33), (4, 16, 17)])
@pytest.mark.push
def test_greater_equal(dims):
    class GreaterEqual(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.greater_equal(a, b)

    inputs = [torch.rand(dims), torch.rand(dims)]

    framework_model = GreaterEqual()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False))


@pytest.mark.push
def test_subtract():
    class Subtract(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a - b

    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]

    framework_model = Subtract()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 32),
        (12, 8640),
    ],
)
@pytest.mark.push
def test_multiply(shape):
    class Multiply(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a * b

    inputs = [torch.rand(shape), torch.rand(shape)]

    framework_model = Multiply()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_remainder():
    class Remainder(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a % b

    inputs = [torch.rand(2, 32, 32), torch.rand(2, 32, 32)]

    framework_model = Remainder()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
