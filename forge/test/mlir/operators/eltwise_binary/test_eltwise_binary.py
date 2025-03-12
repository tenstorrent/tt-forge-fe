# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((10,), 0),
        pytest.param(
            (5, 10),
            1,
            marks=pytest.mark.xfail(
                reason="[run_optimization_graph_passes] RuntimeError: TT_ASSERT @forge/csrc/graph_lib/shape.cpp:135: (i >= 0) && (i < (int)dims_.size())"
            ),
        ),
        pytest.param(
            (3, 5, 10),
            2,
            marks=pytest.mark.xfail(
                reason="[run_optimization_graph_passes] RuntimeError: TT_ASSERT @forge/csrc/graph_lib/shape.cpp:135: (i >= 0) && (i < (int)dims_.size())"
            ),
        ),
        pytest.param(
            (2, 3, 5, 10),
            3,
            marks=pytest.mark.xfail(
                reason="[run_optimization_graph_passes] RuntimeError: TT_ASSERT @forge/csrc/graph_lib/shape.cpp:135: (i >= 0) && (i < (int)dims_.size())"
            ),
        ),
        pytest.param(
            (1, 6, 20, 50, 64),
            4,
            marks=pytest.mark.xfail(
                reason="[run_optimization_graph_passes] RuntimeError: TT_ASSERT @forge/csrc/graph_lib/shape.cpp:135: (i >= 0) && (i < (int)dims_.size())"
            ),
        ),
    ],
)
@pytest.mark.push
def test_stack_and_view(forge_property_recorder, shape, dim):

    class stack_and_view(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x, y):
            stacked = torch.stack((x, y), dim=self.dim)
            new_shape = list(x.shape)
            new_shape[self.dim] *= 2
            return stacked.view(*new_shape)

    x = torch.rand(shape)
    y = torch.rand(shape)

    inputs = [x, y]

    framework_model = stack_and_view(dim)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "einsum_pattern, shape_1, shape_2",
    [
        pytest.param(
            "bqnc,bnchw->bqnhw",
            (1, 100, 8, 32),
            (1, 8, 32, 14, 20),
            marks=pytest.mark.xfail(reason="Tensor mismatch. PCC = 0.0451398042494029"),
        ),
        pytest.param(
            "bqnc,bnchw->bqnhw",
            (3, 99, 7, 31),
            (3, 7, 31, 15, 19),
            marks=pytest.mark.xfail(reason="Tensor mismatch. PCC = 0.008232882538975006"),
        ),
    ],
)
@pytest.mark.push
def test_einsum(forge_property_recorder, einsum_pattern, shape_1, shape_2):
    class EinsumModel(torch.nn.Module):
        def __init__(self, pattern):
            super().__init__()
            self.pattern = pattern

        def forward(self, input_1, input_2):
            return torch.einsum(self.pattern, input_1, input_2)

    input_1 = torch.randn(*shape_1)
    input_2 = torch.randn(*shape_2)
    inputs = [input_1, input_2]

    framework_model = EinsumModel(einsum_pattern)
    framework_model.eval()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_atan2(forge_property_recorder, shape):
    class Atan2(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return torch.atan2(x2, x1)

    inputs = [torch.randn(shape), torch.randn(shape)]
    framework_model = Atan2()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_less(forge_property_recorder, shape_x, shape_y):
    class Less(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.less(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = Less()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False), forge_property_handler=forge_property_recorder)


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
def test_greater(forge_property_recorder, shape_x, shape_y):
    class Greater(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.greater(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = Greater()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False), forge_property_handler=forge_property_recorder)


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
def test_not_equal(forge_property_recorder, shape_x, shape_y):
    class NotEqual(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.ne(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = NotEqual()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False), forge_property_handler=forge_property_recorder)


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
def test_equal(forge_property_recorder, shape):
    class Equal(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.eq(x, y)

    x = torch.rand(shape)
    y = x * 2.0

    inputs = [x, y]

    framework_model = Equal()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False), forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape_dtype",
    [
        ((2, 32, 32), torch.float32),
        ((1, 128), torch.int64),
    ],
)
@pytest.mark.push
def test_add(forge_property_recorder, shape_dtype):
    shape, dtype = shape_dtype

    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    # Generate random tensors of the appropriate shape and dtype
    a = torch.rand(size=shape).to(dtype)
    b = torch.rand(size=shape).to(dtype)

    inputs = [a, b]

    framework_model = Add()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("dims", [(1, 32, 64), (6, 33), (4, 16, 17)])
@pytest.mark.push
def test_greater_equal(forge_property_recorder, dims):
    class GreaterEqual(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.greater_equal(a, b)

    inputs = [torch.rand(dims), torch.rand(dims)]

    framework_model = GreaterEqual()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False), forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_subtract(forge_property_recorder):
    class Subtract(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a - b

    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]

    framework_model = Subtract()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_dtype=False), forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 32),
        (12, 8640),
    ],
)
@pytest.mark.push
def test_multiply(forge_property_recorder, shape):
    class Multiply(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a * b

    inputs = [torch.rand(shape), torch.rand(shape)]

    framework_model = Multiply()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_remainder(forge_property_recorder):
    class Remainder(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a % b

    inputs = [torch.rand(2, 32, 32), torch.rand(2, 32, 32)]

    framework_model = Remainder()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
