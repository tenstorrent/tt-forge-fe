# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "input_shape, scale_factor",
    [
        pytest.param(
            (1, 256, 4, 128),
            2,
            marks=pytest.mark.xfail(
                reason="Unsupported operations while lowering from TTForge to TTIR in forward graph - narrow, pad_tile, sparse_matmul, vstack"
            ),
        ),
        pytest.param(
            (3, 32, 10, 10),
            4,
            marks=pytest.mark.xfail(reason="NotImplementedError: Pixel shuffle decomposition only supports r=2"),
        ),
        pytest.param(
            (2, 98, 6, 6),
            7,
            marks=pytest.mark.xfail(reason="NotImplementedError: Pixel shuffle decomposition only supports r=2"),
        ),
        pytest.param(
            (4, 18, 8, 8),
            3,
            marks=pytest.mark.xfail(reason="NotImplementedError: Pixel shuffle decomposition only supports r=2"),
        ),
        pytest.param(
            (2, 50, 12, 12),
            5,
            marks=pytest.mark.xfail(reason="NotImplementedError: Pixel shuffle decomposition only supports r=2"),
        ),
    ],
)
@pytest.mark.push
def test_pixel_shuffle(forge_property_recorder, input_shape, scale_factor):
    class PixelShuffleModel(nn.Module):
        def __init__(self, scale_factor):
            super().__init__()
            self.model = nn.PixelShuffle(scale_factor)

        def forward(self, x):
            return self.model(x)

    inputs = [torch.randn(*input_shape)]
    framework_model = PixelShuffleModel(scale_factor)
    framework_model.eval()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(
            (1, 256, 6, 6),
            torch.float32,
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
        pytest.param(
            (512,),
            torch.float16,
            marks=pytest.mark.xfail(
                reason="Stage optimized_graph: Data mismatch detected. Issue: https://github.com/tenstorrent/tt-forge-fe/issues/1423 "
            ),
        ),
        pytest.param(
            (224, 224), torch.float32, marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath")
        ),
        pytest.param(
            (1, 8, 224, 224),
            torch.float32,
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
        pytest.param(
            (2, 128, 8, 8),
            torch.float16,
            marks=pytest.mark.xfail(
                reason="Stage optimized_graph: Data mismatch detected. Issue: https://github.com/tenstorrent/tt-forge-fe/issues/1423 "
            ),
        ),
        pytest.param(
            (4, 1, 32, 32),
            torch.float32,
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
        pytest.param(
            (6, 1, 900, 256),
            torch.float16,
            marks=pytest.mark.xfail(
                reason="Stage optimized_graph: Data mismatch detected. Issue: https://github.com/tenstorrent/tt-forge-fe/issues/1423 "
            ),
        ),
        pytest.param(
            (8, 64, 32, 32, 45),
            torch.float16,
            marks=pytest.mark.xfail(
                reason="Stage optimized_graph: Data mismatch detected. Issue: https://github.com/tenstorrent/tt-forge-fe/issues/1423 "
            ),
        ),
    ],
)
@pytest.mark.push
def test_nan_to_num(forge_property_recorder, shape, dtype):
    class nan_to_num(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.nan_to_num(x1)

    compiler_cfg = forge.config.CompilerConfig()

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
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, forge_property_handler=forge_property_recorder
    )
    if dtype == torch.float32:
        verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_isnan(forge_property_recorder, shape, dtype):
    class isnan(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.isnan(x1)

    inputs = [torch.randn(shape, dtype=dtype)]
    mask_nan = torch.rand(shape) < 0.1
    inputs[0][mask_nan] = float("nan")

    framework_model = isnan()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - Atan"
)
@pytest.mark.parametrize(
    "shape",
    [(888), (1, 7, 256), (3, 128, 128), (1, 10), (2, 2, 2), (5, 5), (1, 3, 224, 224), (8, 16, 32), (1, 3, 2, 544, 544)],
)
@pytest.mark.push
def test_atan(forge_property_recorder, shape):
    class Atan(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.atan(x1)

    inputs = [torch.randn(shape)]

    framework_model = Atan()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_power(forge_property_recorder, shape):
    class power(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.pow(x, 0.75)

    inputs = [torch.rand(shape)]

    framework_model = power()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 7, 256),
    ],
)
@pytest.mark.push
def test_sin(forge_property_recorder, shape):
    class Sin(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sin(x)

    inputs = [torch.rand(shape)]

    framework_model = Sin()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 7, 256),
    ],
)
@pytest.mark.push
def test_cosine(forge_property_recorder, shape):
    class Cosine(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cos(x)

    inputs = [torch.rand(shape)]

    framework_model = Cosine()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 768),
    ],
)
@pytest.mark.xfail(reason="Found Unsupported operations while lowering from TTForge to TTIR in forward graph")
@pytest.mark.push
def test_tanh(forge_property_recorder, shape):
    class Tanh(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tanh(x)

    inputs = [torch.rand(shape)]

    framework_model = Tanh()

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 512, 512),
    ],
)
@pytest.mark.push
def test_leakyrelu(forge_property_recorder, shape):

    inputs = [torch.rand(shape)]

    framework_model = nn.LeakyReLU(negative_slope=0.1)

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 4096),
    ],
)
@pytest.mark.push
def test_gelu(forge_property_recorder, shape):

    inputs = [torch.rand(shape)]

    framework_model = nn.GELU()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_clip(forge_property_recorder, shape, min_val, max_val):
    class Clip(nn.Module):
        def __init__(self, min_val, max_val):
            super().__init__()
            self.min_val = min_val
            self.max_val = max_val

        def forward(self, x):
            return torch.clamp(x, self.min_val, self.max_val)

    inputs = [torch.rand(shape)]

    framework_model = Clip(min_val, max_val)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((56), 0),
        ((1, 128), 1),
        pytest.param(
            (1, 64, 76),
            2,
            marks=pytest.mark.xfail(reason="ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)"),
        ),
        pytest.param(
            (1, 64, 76, 96),
            3,
            marks=pytest.mark.xfail(reason="ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)"),
        ),
        pytest.param(
            (1, 64, 86, 100, 120),
            4,
            marks=pytest.mark.xfail(
                reason=" RuntimeError: (dim >= 0 && dim <= 3),info: dim should be 0 - 3, but got: 4"
            ),
        ),
    ],
)
@pytest.mark.push
def test_cumsum(forge_property_recorder, shape, dim):
    class CumSum(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.cumsum(x, dim=self.dim)

    inputs = [torch.rand(shape)]

    framework_model = CumSum(dim)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape", [(1, 1, 256, 256), (1, 1, 1, 128), (1, 1, 1, 384), (1, 1, 32, 32), (1, 1, 6, 6), (1, 1, 29, 29)]
)
@pytest.mark.push
def test_abs(forge_property_recorder, shape):
    class Abs(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.abs(x)

    inputs = [torch.rand(shape)]

    framework_model = Abs()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_exp(forge_property_recorder, shape):
    class Exp(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(x)

    inputs = [torch.rand(shape)]

    framework_model = Exp()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_log(forge_property_recorder, shape):
    class Log(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.log(x)

    inputs = [torch.rand(shape)]

    framework_model = Log()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_maximum(forge_property_recorder, shape_x, shape_y):
    class Maximum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.maximum(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    inputs = [x, y]

    framework_model = Maximum()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_relu(forge_property_recorder):
    class ReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, a):
            return self.relu(a)

    inputs = [torch.rand(1, 32)]

    framework_model = ReLU()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("x_shape", [7, 32, 41])
@pytest.mark.parametrize("y_shape", [7, 32, 41])
@pytest.mark.push
def test_sqrt(forge_property_recorder, x_shape, y_shape):
    class Sqrt(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sqrt(x)

    inputs = [
        torch.rand(1, x_shape, y_shape),
    ]

    framework_model = Sqrt()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_reciprocal(forge_property_recorder, shape):
    class Reciprocal(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reciprocal(x)

    inputs = [
        torch.rand(*shape) + 0.1,  # Adding 0.1 to avoid division by zero
    ]

    framework_model = Reciprocal()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_sigmoid(forge_property_recorder, shape):
    class Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sigmoid(x)

    inputs = [
        torch.rand(*shape),
    ]
    framework_model = Sigmoid()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 768),
    ],
)
@pytest.mark.push
def test_tanh(forge_property_recorder, input_shape):
    class Tanh(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.tanh(a)

    inputs = [torch.rand(input_shape)]

    framework_model = Tanh()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_floor(forge_property_recorder, input_data):
    class Floor(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.floor(a)

    framework_model = Floor()
    compiled_model = forge.compile(
        framework_model, sample_inputs=[input_data], module_name="floor", forge_property_handler=forge_property_recorder
    )

    verify([input_data], framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape, dim, keepdim",
    [
        # Core test cases for dimension-specific argmax
        pytest.param(
            (56,),
            0,
            False,
            marks=pytest.mark.xfail(
                reason="This argmax reduction should return a scalar, but that's not supported yet"
            ),
        ),
        ((56,), 0, True),
        ((1, 128), 1, False),
        ((1, 128), 1, True),
        pytest.param(
            (1, 64, 76),
            2,
            False,
            marks=pytest.mark.xfail(reason="ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)"),
        ),
        ((1, 64, 76), 2, True),
        pytest.param(
            (1, 64, 76), 1, True, marks=pytest.mark.xfail(reason="TTNN: Only argmax on last dim is supported!")
        ),
        # Core test cases for global argmax (dim=None)
        pytest.param(
            (56,),
            None,
            False,
            marks=pytest.mark.xfail(
                reason="This argmax reduction should return a scalar, but that's not supported yet"
            ),
        ),
        ((56,), None, True),
        pytest.param(
            (1, 128),
            None,
            False,
            marks=pytest.mark.xfail(
                reason="This argmax reduction should return a scalar, but that's not supported yet"
            ),
        ),
        ((1, 128), None, True),
    ],
)
@pytest.mark.push
def test_argmax(forge_property_recorder, shape, dim, keepdim):
    class ArgMax(nn.Module):
        def __init__(self, dim, keepdim):
            super().__init__()
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, x):
            return torch.argmax(x, dim=self.dim, keepdim=self.keepdim)

    inputs = [torch.rand(shape)]

    framework_model = ArgMax(dim, keepdim)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
