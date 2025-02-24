# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(
            (1, 256, 6, 6),
            torch.float32,
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
        pytest.param((512,), torch.float16),
        pytest.param(
            (224, 224), torch.float32, marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath")
        ),
        pytest.param(
            (1, 8, 224, 224),
            torch.float32,
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
        pytest.param((2, 128, 8, 8), torch.float16),
        pytest.param(
            (4, 1, 32, 32),
            torch.float32,
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
        pytest.param((6, 1, 900, 256), torch.float16),
        pytest.param((8, 64, 32, 32, 45), torch.float16),
    ],
)
@pytest.mark.push
def test_nan_to_num(shape, dtype):
    class nan_to_num(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.nan_to_num(x1)

    compiler_cfg = forge.config._get_global_compiler_config()

    if dtype == torch.float16:
        # set compie depth to avoid Unsupported ttnn::DataType , Fatal Python error: Aborted
        compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    inputs = [torch.randn(shape, dtype=dtype)]

    mask_nan = torch.rand(shape, dtype=dtype) < 0.1
    mask_posinf = torch.rand(shape, dtype=dtype) < 0.05
    mask_neginf = torch.rand(shape, dtype=dtype) < 0.05

    inputs[0][mask_nan] = float("nan")
    inputs[0][mask_posinf] = float("inf")
    inputs[0][mask_neginf] = float("-inf")

    framework_model = nan_to_num()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    if dtype == torch.float32:
        verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(
            (1, 256),
            torch.float32,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.float32"
            ),
        ),
        pytest.param(
            (1, 512, 14, 14),
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.bfloat16"
            ),
        ),
        pytest.param(
            (1, 3, 224),
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.bfloat16"
            ),
        ),
        pytest.param(
            (1, 8, 224, 224),
            torch.float32,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.float32"
            ),
        ),
        pytest.param(
            (1, 8, 16, 128, 128),
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.bfloat16"
            ),
        ),
        pytest.param(
            (4, 1, 32, 32),
            torch.float32,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.float32"
            ),
        ),
        pytest.param(
            (100,),
            torch.float32,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.float32"
            ),
        ),
        pytest.param(
            (1, 8, 64, 32, 32),
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="Dtype mismatch: framework_model.dtype=torch.bool, compiled_model.dtype=torch.bfloat16"
            ),
        ),
    ],
)
@pytest.mark.push
def test_isnan(shape, dtype):
    class isnan(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.isnan(x1)

    inputs = [torch.randn(shape, dtype=dtype)]
    mask_nan = torch.rand(shape) < 0.1
    inputs[0][mask_nan] = float("nan")

    framework_model = isnan()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - Atan"
)
@pytest.mark.parametrize(
    "shape",
    [(888), (1, 7, 256), (3, 128, 128), (1, 10), (2, 2, 2), (5, 5), (1, 3, 224, 224), (8, 16, 32), (1, 3, 2, 544, 544)],
)
@pytest.mark.push
def test_atan(shape):
    class Atan(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.atan(x1)

    inputs = [torch.randn(shape)]

    framework_model = Atan()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (1, 96),
        (10, 10),
        (1, 96, 54),
        (1, 64, 128),
        (1, 96, 54, 54),
        (1, 3, 224, 224),
        (1, 64, 128, 128),
        (1, 96, 28, 28),
        (1, 1, 128, 128),
    ],
)
@pytest.mark.push
def test_power(shape):
    class power(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.pow(x, 0.75)

    inputs = [torch.rand(shape)]

    framework_model = power()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 7, 256),
    ],
)
@pytest.mark.push
def test_sin(shape):
    class Sin(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sin(x)

    inputs = [torch.rand(shape)]

    framework_model = Sin()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 7, 256),
    ],
)
@pytest.mark.push
def test_cosine(shape):
    class Cosine(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cos(x)

    inputs = [torch.rand(shape)]

    framework_model = Cosine()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 768),
    ],
)
@pytest.mark.xfail(reason="Found Unsupported operations while lowering from TTForge to TTIR in forward graph")
@pytest.mark.push
def test_tanh(shape):
    class Tanh(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tanh(x)

    inputs = [torch.rand(shape)]

    framework_model = Tanh()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 512, 512),
    ],
)
@pytest.mark.push
def test_leakyrelu(shape):

    inputs = [torch.rand(shape)]

    framework_model = nn.LeakyReLU(negative_slope=0.1)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 4096),
    ],
)
@pytest.mark.push
def test_gelu(shape):

    inputs = [torch.rand(shape)]

    framework_model = nn.GELU()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape, min_val, max_val",
    [
        ((1, 1, 256, 256), 0, 1),
        ((1, 96, 96, 24), 6.0, 0.0),
        ((1, 1, 32, 32), -0.5, 0.5),
        ((2, 10, 5, 20), 2.0, -1.0),
        ((3, 3, 3, 3), -3.0, -1.0),
        ((1, 64, 64), -0.5, 0.5),
        ((1, 128, 128), 1.0, -1.0),
        ((2, 2, 2), -1.0, 0.0),
        ((32, 32), -0.2, 0.2),
        ((3, 3), -0.5, -0.2),
        ((4,), 0.0, 2.0),
        ((8,), -3.0, -1.0),
    ],
)
@pytest.mark.push
def test_clip(shape, min_val, max_val):
    class Clip(nn.Module):
        def __init__(self, min_val, max_val):
            super().__init__()
            self.min_val = min_val
            self.max_val = max_val

        def forward(self, x):
            return torch.clamp(x, self.min_val, self.max_val)

    inputs = [torch.rand(shape)]

    framework_model = Clip(min_val, max_val)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((1, 128), 1),
    ],
)
@pytest.mark.xfail(reason="Found Unsupported operations while lowering from TTForge to TTIR in forward graph")
@pytest.mark.push
def test_cumsum(shape, dim):
    class CumSum(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.cumsum(x, dim=self.dim)

    inputs = [torch.rand(shape)]

    framework_model = CumSum(dim)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape", [(1, 1, 256, 256), (1, 1, 1, 128), (1, 1, 1, 384), (1, 1, 32, 32), (1, 1, 6, 6), (1, 1, 29, 29)]
)
@pytest.mark.push
def test_abs(shape):
    class Abs(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.abs(x)

    inputs = [torch.rand(shape)]

    framework_model = Abs()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    ],
)
@pytest.mark.push
def test_exp(shape):
    class Exp(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(x)

    inputs = [torch.rand(shape)]

    framework_model = Exp()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 28, 28),
        (128, 28, 28),
        (28, 28),
        (28),
        (1, 64, 28, 28),
        (64, 28, 28),
        (1, 256, 28, 28),
        (256, 28, 28),
        (1, 128, 14, 14),
        (128, 14, 14),
        (14, 14),
        (14),
        (1, 128, 56, 56),
        (128, 56, 56),
        (56, 56),
        (56),
        (1, 32, 64, 64),
        (32, 64, 64),
        (64, 64),
        (64),
        (1, 512, 7, 7),
        (512, 7, 7),
        (7, 7),
        (7),
    ],
)
@pytest.mark.push
def test_log(shape):
    class Log(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.log(x)

    inputs = [torch.rand(shape)]

    framework_model = Log()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape_x, shape_y",
    [
        ((1, 12, 256, 256), (1,)),
        ((1, 16, 256, 256), (1,)),
        ((1, 32, 256, 256), (1,)),
        ((1, 12, 32, 32), (1,)),
        ((1, 16, 32, 32), (1,)),
        ((1, 32, 32, 32), (1,)),
    ],
)
@pytest.mark.xfail(
    reason="TTNN maximum op: unsupported broadcast. Tracking on: https://github.com/tenstorrent/tt-metal/issues/16969"
)
@pytest.mark.push
def test_maximum(shape_x, shape_y):
    class Maximum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.maximum(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = Maximum()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_relu():
    class ReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, a):
            return self.relu(a)

    inputs = [torch.rand(1, 32)]

    framework_model = ReLU()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("x_shape", [7, 32, 41])
@pytest.mark.parametrize("y_shape", [7, 32, 41])
@pytest.mark.push
def test_sqrt(x_shape, y_shape):
    class Sqrt(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sqrt(x)

    inputs = [
        torch.rand(1, x_shape, y_shape),
    ]

    framework_model = Sqrt()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (7,),  # 1D tensor
        (32,),  # 1D tensor
        (7, 32),  # 2D tensor
        (32, 41),  # 2D tensor
        (1, 7, 32),  # 3D tensor
        (1, 32, 41),  # 3D tensor
        (1, 7, 32, 41),  # 4D tensor
        (2, 7, 32, 41),  # 4D tensor
    ],
)
@pytest.mark.push
def test_reciprocal(shape):
    class Reciprocal(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reciprocal(x)

    inputs = [
        torch.rand(*shape) + 0.1,  # Adding 0.1 to avoid division by zero
    ]

    framework_model = Reciprocal()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape",
    [
        (7,),  # 1D tensor
        (32,),  # 1D tensor
        (7, 32),  # 2D tensor
        (32, 41),  # 2D tensor
        (1, 7, 32),  # 3D tensor
        (1, 32, 41),  # 3D tensor
        (1, 7, 32, 41),  # 4D tensor
        (2, 7, 32, 41),  # 4D tensor
    ],
)
@pytest.mark.push
def test_sigmoid(shape):
    class Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sigmoid(x)

    inputs = [
        torch.rand(*shape),
    ]
    framework_model = Sigmoid()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 768),
    ],
)
@pytest.mark.push
def test_tanh(input_shape):
    class Tanh(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.tanh(a)

    inputs = [torch.rand(input_shape)]

    framework_model = Tanh()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(reason="RuntimeError: BinaryOpType cannot be mapped to BcastOpMath")
@pytest.mark.parametrize(
    "input_data",
    [
        torch.tensor([-0.8166, 1.5308, -0.2530, -0.2091]),
        torch.tensor([-3.7, -1.2, 0.0, 1.5, 3.9]),
        torch.tensor([1.0, 2.0, -1.0, -2.0]),
        torch.tensor([-12345.678, 12345.678, -0.999, 0.999, 3.14159, -3.14159]),
    ],
)
def test_floor(input_data):
    class Floor(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.floor(a)

    framework_model = Floor()
    compiled_model = forge.compile(framework_model, sample_inputs=[input_data], module_name="floor")

    verify([input_data], framework_model, compiled_model)
