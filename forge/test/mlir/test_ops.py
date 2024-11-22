# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import pytest
import torch
from torch import nn, threshold

import forge
from forge.op.eval.common import compare_with_golden_pcc, compare_with_golden
from forge.tensor import to_forge_tensors, to_pt_tensors


@pytest.mark.parametrize(
    "shape, kernel_size, stride",
    [
        ((1, 1, 100, 54, 54), (5, 1, 1), (1, 1, 1)),
        ((1, 2, 5, 5, 5), (3, 3, 3), (2, 2, 2)),
        ((1, 4, 100, 54, 54), (3, 1, 1), (1, 1, 1)),
        ((1, 8, 32, 16, 16), (4, 1, 1), (1, 1, 1)),
        ((1, 1, 100, 54, 54), (5, 1, 1), (5, 1, 1)),
        ((1, 4, 10, 4, 4), (1, 1, 1), (1, 1, 1)),
        ((1, 16, 32, 16, 16), (8, 1, 1), (3, 3, 3)),
    ],
)
@pytest.mark.push
def test_avgpool3d(shape, kernel_size, stride):
    class AvgPool3D(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return nn.functional.avg_pool3d(x, kernel_size=kernel_size, stride=stride)

    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = (
        forge.CompileDepth.SPLIT_GRAPH
    )  # Due to #https://github.com/tenstorrent/tt-mlir/issues/1343
    inputs = [torch.rand(shape)]

    framework_model = AvgPool3D()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    if compiler_cfg.compile_depth == forge.CompileDepth.FULL:
        co_out = compiled_model(*inputs)
        co_out = [co.to("cpu") for co in co_out]
        fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
        assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride_size, padding, ceil_mode",
    [
        pytest.param(
            (1, 96, 54, 54),
            3,
            2,
            0,
            True,
            marks=pytest.mark.xfail(
                reason="Runtime Error  : Invalid sharding configuration: For Row Major layout with element size of 2 bytes, the innermost dimension must align to 2 bytes."
            ),
        ),
        pytest.param(
            (1, 64, 55, 54),
            3,
            2,
            0,
            True,
            marks=pytest.mark.xfail(
                reason="Runtime Error  : Invalid sharding configuration: For Row Major layout with element size of 2 bytes, the innermost dimension must align to 2 bytes."
            ),
        ),
        pytest.param(
            (1, 128, 26, 26),
            3,
            2,
            0,
            True,
            marks=pytest.mark.xfail(
                reason="Runtime Error  : Invalid sharding configuration: For Row Major layout with element size of 2 bytes, the innermost dimension must align to 2 bytes."
            ),
        ),
        pytest.param(
            (1, 256, 26, 26),
            3,
            2,
            0,
            True,
            marks=pytest.mark.xfail(
                reason="Runtime Error  : Invalid sharding configuration: For Row Major layout with element size of 2 bytes, the innermost dimension must align to 2 bytes."
            ),
        ),
        pytest.param(
            (1, 96, 54, 54),
            3,
            2,
            0,
            False,
            marks=pytest.mark.xfail(reason="Runtime Error  : Shard page size must currently have L1 aligned page size"),
        ),
        pytest.param(
            (1, 64, 55, 54),
            3,
            2,
            0,
            False,
            marks=pytest.mark.xfail(
                reason="Runtime Error  : Shard page size must currently have L1 aligned page size."
            ),
        ),
        pytest.param(
            (1, 128, 26, 26),
            3,
            2,
            0,
            False,
            marks=pytest.mark.xfail(reason="Runtime Error  : Shard page size must currently have L1 aligned page size"),
        ),
        pytest.param(
            (1, 256, 26, 26),
            3,
            2,
            0,
            False,
            marks=pytest.mark.xfail(reason="Runtime Error  : Shard page size must currently have L1 aligned page size"),
        ),
        pytest.param(
            (1, 3, 32, 32),
            3,
            3,
            (1, 1, 1, 1),
            False,
            marks=pytest.mark.xfail(
                reason="Invalid sharding configuration: For Row Major layout with element size of 2 bytes, the innermost dimension must align to 2 bytes"
            ),
        ),
        pytest.param(
            (1, 3, 32, 32),
            3,
            3,
            (1, 1, 2, 2),
            False,
            marks=pytest.mark.xfail(
                reason="Invalid sharding configuration: For Row Major layout with element size of 2 bytes, the innermost dimension must align to 2 bytes"
            ),
        ),
    ],
)
@pytest.mark.push
def test_maxpool2d(input_shape, kernel_size, stride_size, padding, ceil_mode):
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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "shape, mode",
    [
        ((1, 2048, 7, 7), "nearest"),
        ((1, 2048, 7, 7), "bilinear"),
    ],
)
@pytest.mark.xfail(reason="Found Unsupported operations while lowering from TTForge to TTIR in forward graph")
@pytest.mark.push
def test_interpolate(shape, mode):
    class Interpolate(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return nn.functional.interpolate(x, scale_factor=2, mode=mode)

    inputs = [torch.rand(shape)]

    framework_model = Interpolate()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "shape",
    [
        (1, 256, 6, 6),
        (1, 3, 64, 64),
        (1, 512, 14, 14),
        (1, 3, 224, 224),
        (2, 256, 10, 10),
        (1, 512, 3, 3),
        (1, 1000, 1, 1),
        (2, 128, 8, 8),
        (4, 1, 32, 32),
        (8, 64, 32, 32),
    ],
)
@pytest.mark.push
def test_flatten(shape):
    class Flatten(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.flatten(x, 1)

    inputs = [torch.rand(shape)]

    framework_model = Flatten()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize("operand_and_cast_dtype", [(torch.float32, torch.int32), (torch.int32, torch.float32)])
@pytest.mark.push
def test_cast(operand_and_cast_dtype):

    operand_dtype = operand_and_cast_dtype[0]
    cast_dtype = operand_and_cast_dtype[1]

    class Cast(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return a.to(cast_dtype)

    def get_input_tensor(dtype):
        shape = (1, 32, 32)
        if dtype in [torch.float32, torch.bfloat16]:
            return torch.rand(shape, dtype=dtype)
        elif dtype in [torch.int32]:
            return torch.randint(high=torch.iinfo(dtype).max, size=shape, dtype=dtype)
        else:
            raise Exception("Unsupported datatype")

    inputs = [
        get_input_tensor(operand_dtype),
    ]

    framework_model = Cast()
    framework_model.eval()

    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([fo.dtype == co.dtype for fo, co in zip(fw_out, co_out)])
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "batch_size, num_channels, height, width",
    [
        (1, 32, 56, 56),
    ],
)
@pytest.mark.xfail(reason="shape mismatch: expected [1], got []")
@pytest.mark.push
def test_layernorm(batch_size, num_channels, height, width):

    # framework_model = nn.LayerNorm((num_channels, height, width)) # Support only normalization over last one dimension
    framework_model = nn.LayerNorm((width))

    inputs = [torch.rand(batch_size, num_channels, height, width)]
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
@pytest.mark.xfail(reason="'ttnn.clamp' op input and output must have same shape")
@pytest.mark.push
def test_clip(shape, min_val, max_val):
    class Clip(nn.Module):
        def __init__(self, min_val, max_val):
            super().__init__()
            self.min_val = min_val
            self.max_val = max_val

        def forward(self, x):
            return torch.clamp(x, self.min_val, self.max_val)

    framework_model = Clip(min_val, max_val)
    inputs = [torch.rand(shape)]

    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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

    framework_model = CumSum(dim)
    inputs = [torch.rand(shape)]

    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "condition, input, other",
    [
        (
            [[1, 0], [0, 1]],
            [[1, 2], [3, 4]],
            [[10, 20], [30, 40]],
        ),
    ],
)
@pytest.mark.xfail(reason="Unsupported data format during lowering from TTForge to TTIR: Bfp2_b")
@pytest.mark.push
def test_where(condition, input, other):
    class Where(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, condition, input1, input2):
            return torch.where(condition, input1, input2)

    condition = torch.tensor(condition, dtype=torch.bool)
    input = torch.tensor(input)
    other = torch.tensor(other)

    framework_model = Where()

    inputs = [condition, input, other]

    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
@pytest.mark.xfail(reason="TTNN maximum op: unsupported broadcast")
@pytest.mark.push
def test_maximum(shape_x, shape_y):
    class Maximum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.maximum(x, y)

    x = torch.rand(shape_x)
    y = torch.rand(shape_y)

    framework_model = Maximum()
    inputs = [x, y]

    fw_out = framework_model(*inputs)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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

    framework_model = Less()
    inputs = [x, y]

    fw_out = framework_model(*inputs)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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

    framework_model = Greater()
    inputs = [x, y]

    fw_out = framework_model(*inputs)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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

    framework_model = NotEqual()
    inputs = [x, y]

    fw_out = framework_model(*inputs)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden(golden=fo, calculated=co) for fo, co in zip(fw_out, co_out)])


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

    framework_model = Equal()
    inputs = [x, y]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    fw_out = framework_model(*inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden(golden=fo, calculated=co) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "batch_size, num_channels, height, width",
    [
        (1, 32, 56, 56),
        (1, 64, 112, 112),
        (1, 128, 224, 224),
        (1, 256, 28, 28),
        (1, 32, 56, 56),
        (2, 64, 112, 112),  # pcc = 0.6199620538910243
        (4, 64, 28, 28),  # pcc = 0.4935656199688308
        (8, 64, 112, 112),  # pcc = 0.40443518583193394
        (16, 128, 224, 224),  # pcc = -0.0004391043640747615
        (32, 256, 28, 28),  # pcc = 0.39200606381500713
    ],
)
@pytest.mark.push
def test_batchnorm2d(batch_size, num_channels, height, width):

    framework_model = nn.BatchNorm2d(num_features=num_channels)

    inputs = [torch.rand(batch_size, num_channels, height, width)]
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    if batch_size == 1:
        assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.push
def test_add():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    inputs = [torch.rand(2, 32, 32), torch.rand(2, 32, 32)]

    framework_model = Add()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


params = [
    ((1, 32, 64), (-1, -2)),
    ((1, 64, 32), (1, 2)),
    ((1, 32, 64, 128), (3, 2)),
    ((32, 128), (0, 1)),
    ((18, 65), (1, 0)),
    ((6, 33, 34), (-1, 1)),
    ((1, 32, 64), (-2, -3)),
    ((6, 33, 34), (-1, -3)),
    ((32, 128, 24), (1, -3)),
    ((1, 12, 32, 100), (-3, -2)),
    ((32, 12, 100), (-1, -2)),
]
# Dynamically generate params with conditional xfail
param_list = []
for param in params:
    for data_format in [torch.float32, torch.bfloat16]:
        if data_format == torch.bfloat16 and param in [
            ((18, 65), (1, 0)),
            ((6, 33, 34), (-1, 1)),
            ((6, 33, 34), (-1, -3)),
            ((32, 128, 24), (1, -3)),
            ((1, 12, 32, 100), (-3, -2)),
            ((32, 12, 100), (-1, -2)),
        ]:
            param_list.append(
                pytest.param(
                    param,
                    data_format,
                    marks=pytest.mark.xfail(
                        reason="Tensor mismatch issue for bfloat16. Metal tracking issue: https://github.com/tenstorrent/tt-metal/issues/15099"
                    ),
                )
            )
        else:
            param_list.append((param, data_format))


@pytest.mark.parametrize("params, data_format", param_list)
@pytest.mark.push
def test_transpose(params, data_format):
    class Transpose(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, a):
            return torch.transpose(a, *self.dims)

    input_shape, dims = params
    inputs = [torch.rand(input_shape, dtype=data_format)]  # Use data_format instead of hardcoded dtype
    # Initialize the model with data_formats
    framework_model = Transpose(dims).to(data_format)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "source_and_target_shape",
    [((8, 32, 256), (2, 4, 32, 256)), ((8, 32, 32), (1, 2, 4, 32, 32)), ((8192, 128), (1, 256, 32, 128))],
    ids=["1", "2", "3"],
)
@pytest.mark.push
def test_reshape(source_and_target_shape):
    source_shape, target_shape = source_and_target_shape

    class Reshape(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.reshape(a, target_shape)

    inputs = [torch.rand(source_shape)]
    framework_model = Reshape()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((1, 8, 16, 32, 32), 0),
        ((8, 1, 16, 32, 32), 1),
        ((8, 16, 1, 32, 32), 2),
        ((1, 8, 16, 32, 32), -5),
        ((8, 1, 16, 32, 32), -4),
        ((8, 16, 1, 32, 32), -3),
        ([1, 12, 3200], 0),
        ([1, 1, 2048, 1], [-3, -4]),
        ([1, 64, 1, 1], [-1, -4]),
        ([1, 1, 1, 128], [-2, -4]),
        ([1, 1, 32, 1], [-1, -3]),
        ([1, 1, 1, 64], [-4, -3]),
    ],
)
@pytest.mark.push
def test_squeeze(input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    if input_shape == [1, 12, 3200] or isinstance(dim, list) and len(dim) > 1 and all(d < 0 for d in dim):
        pytest.xfail("TTNN: Tensor layout issues with non tile dim aligned shapes")

    class Squeeze(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.squeeze(a, dim)

    inputs = [torch.rand(*input_shape)]

    framework_model = Squeeze()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((8, 16, 32, 32), 0),
        ((8, 16, 32, 32), 1),
        ((8, 16, 32, 32), 2),
        ((8, 16, 32, 32), -3),
        ((8, 16, 32, 32), -4),
        ([12, 8640], 0),
    ],
)
@pytest.mark.push
def test_unsqueeze(input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    if input_shape == [12, 8640]:
        pytest.xfail("TTNN: Tensor layout issues with non tile dim aligned shapes")

    class Unsqueeze(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.unsqueeze(a, dim)

    inputs = [torch.rand(*input_shape)]

    framework_model = Unsqueeze()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "inputs_and_dim",
    [
        ((2, 2, 32, 32), (2, 2, 32, 32), 0),
        ((2, 2, 32, 32), (2, 2, 32, 32), 1),
        ((2, 2, 32, 32), (2, 2, 32, 32), 2),
        ((2, 2, 32, 32), (2, 2, 32, 32), 3),
        ((2, 2, 32, 32), (2, 2, 32, 32), -1),
        ((2, 2, 32, 32), (2, 2, 32, 32), -2),
        ((2, 2, 32, 32), (2, 2, 32, 32), -3),
        ((2, 2, 32, 32), (2, 2, 32, 32), -4),
    ],
    ids=["0", "1", "2", "3", "-1", "-2", "-3", "-4"],
)
@pytest.mark.push
def test_concat(inputs_and_dim):
    in_shape1, in_shape2, dim = inputs_and_dim

    class Concat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.cat((a, b), dim)

    inputs = [torch.rand(in_shape1), torch.rand(in_shape2)]

    framework_model = Concat()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    output = co_out[0].to(torch.bool)
    assert compare_with_golden(golden=fw_out, calculated=output)


@pytest.mark.push
def test_subtract():
    class Subtract(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a - b

    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]

    framework_model = Subtract()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.skip(reason="This is not ready yet")
@pytest.mark.push
def test_linear():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(20, 30, bias=True)

        def forward(self, a):
            return self.l1(a)

    inputs = [torch.rand(1, 128, 20)]

    framework_model = Linear()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.push
def test_softmax():
    class Softmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, a):
            return self.softmax(a)

    inputs = [torch.rand(1, 128)]

    framework_model = Softmax()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize("input_shape", [(1, 32, 32), (1, 64, 64), (1, 128, 128, 128)], ids=["32", "64", "128"])
@pytest.mark.parametrize("dim", [-1, -2], ids=["-1", "-2"])
@pytest.mark.push
def test_reduce_sum(input_shape, dim):
    class ReduceSum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            # reduce is supported on tt-metal only with keepdim=True
            return torch.sum(a, dim=dim, keepdim=True)

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceSum()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 32, 12),
        (1, 12, 32),
        (1, 12, 3200),
        (1, 32, 32),
        (1, 64, 64),
        (1, 128, 128, 128),
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        -1,
        -2,
    ],
)
@pytest.mark.push
def test_reduce_mean(input_shape, dim):

    if input_shape == (1, 12, 3200) and dim == -1:
        # Tensor mismatch(PCC: 0.72) - https://github.com/tenstorrent/tt-mlir/issues/869
        pytest.xfail("Tensor mismatch between PyTorch and TTNN (PCC: 0.72)")

    class ReduceMean(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            # reduce is supported on tt-metal only with keepdim=True
            return torch.mean(a, dim=dim, keepdim=True)

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceMean()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("batch_size", [1, 7, 32])
@pytest.mark.parametrize("outer_dim_x", [7, 32, 41, 64])
@pytest.mark.parametrize("outer_dim_y", [7, 32, 41, 64])
@pytest.mark.parametrize("inner_dim", [1, 7, 32, 41, 64])
@pytest.mark.push
def test_matmul(batch_size, outer_dim_x, outer_dim_y, inner_dim):
    class Matmul(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    inputs = [
        torch.rand(batch_size, outer_dim_x, inner_dim),
        torch.rand(batch_size, inner_dim, outer_dim_y),
    ]

    framework_model = Matmul()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("x_shape", [7, 32, 41])
@pytest.mark.parametrize("y_shape", [7, 32, 41])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.push
def test_mean(x_shape, y_shape, dim):
    class Mean(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.mean(x, dim=dim)

    inputs = [
        torch.rand(1, x_shape, y_shape),
    ]

    framework_model = Mean()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


# @pytest.mark.parametrize("vocab_size", [2048, 16384, 32000])
# @pytest.mark.parametrize("token_num", [1, 7, 32])
# @pytest.mark.parametrize("embedding_dim", [128, 512, 3200])
@pytest.mark.xfail(reason="ttnn.embedding op fails while reshaping the input_tensor in TILE_LAYOUT")
@pytest.mark.parametrize("vocab_size", [32000])
@pytest.mark.parametrize("token_num", [12])
@pytest.mark.parametrize("embedding_dim", [3200])
@pytest.mark.push
def test_embedding(vocab_size, token_num, embedding_dim):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False

    class Embedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            return self.embedding(x)

    inputs = [
        torch.randint(0, vocab_size, (1, token_num)).to(torch.int32),
    ]

    framework_model = Embedding()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


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
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("dim", [-1, -2, -3], ids=["-1", "-2", "-3"])
@pytest.mark.parametrize("start", [0], ids=["0"])
@pytest.mark.parametrize("stop", [2, 32, 64], ids=["2", "32", "64"])
@pytest.mark.parametrize("stride", [1, 2, 4, 8], ids=["1", "2", "4", "8"])
@pytest.mark.parametrize("shape", [(1, 32, 64, 64), (32, 64, 64), (64, 64)])
@pytest.mark.push
def test_indexing(dim, start, stop, stride, shape):
    if len(shape) == 2 and dim == -3:
        pytest.skip("Skipping since indexing on dim=-3, 2D tensor doesn't make sense")
    if stop > shape[dim]:
        pytest.skip("Skipping since stop > shape[dim]")

    class ForgeIndexing(forge.ForgeModule):
        def __init__(self, dim, start, stop, stride):
            super().__init__("ForgeTest")

        def forward(self, x):
            return forge.op.Index("indexing_op_1", x, dim, start, stop, stride)

    inputs = to_forge_tensors([torch.rand(*shape)])
    model = ForgeIndexing(dim, start, stop, stride)
    golden_out = model(*inputs)

    compiled_model = forge.compile(model, sample_inputs=inputs)

    inputs = to_pt_tensors(inputs)
    compiled_output = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in compiled_output]
    assert compare_with_golden_pcc(golden=golden_out.value(), calculated=co_out[0], pcc=0.99)


@pytest.mark.xfail(reason="ttnn.embedding op fails while reshaping the input_tensor in TILE_LAYOUT")
@pytest.mark.parametrize(
    "indices_shape",
    [
        (12,),
        (32,),
        (1, 7),
        (1, 28),
    ],
)
@pytest.mark.parametrize(
    "input_tensor_shape",
    [
        (12, 100),
        (3200, 512),
        (2048, 128),
        (4127, 256),
    ],
)
@pytest.mark.push
def test_adv_index_embedding_decompostion(indices_shape, input_tensor_shape):
    class ForgeAdvIndex(forge.ForgeModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, input_tensor, indices):
            return forge.op.AdvIndex("adv_index_op_1", input_tensor, indices)

    model = ForgeAdvIndex("ForgeAdvIndex")

    # Sample Inputs
    pt_input_tensor = torch.rand(input_tensor_shape).to(torch.float32)
    pt_indices = torch.randint(input_tensor_shape[0], indices_shape).to(torch.int32)
    inputs = to_forge_tensors([pt_input_tensor, pt_indices])

    # Sanity run
    golden_out = model(*inputs)

    # Compile the model
    compiled_model = forge.compile(model, sample_inputs=inputs)

    # Run on TT device
    inputs = to_pt_tensors(inputs)
    compiled_output = compiled_model(*inputs)
    co_out = [co.to("cpu") for co in compiled_output]

    # Validate results
    assert compare_with_golden_pcc(golden=golden_out.value(), calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 32, 64, 64),
        (3, 22, 37, 41),
        (2, 32, 64),
        (3, 22, 37),
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
        2,
        3,
        -1,
        -2,
        -3,
        -4,
    ],
)
@pytest.mark.push
def test_reduce_max(input_shape, dim):

    reduce_max_dim = dim
    if reduce_max_dim < 0:
        reduce_max_dim = reduce_max_dim + len(input_shape)
    if (reduce_max_dim < 0) or (reduce_max_dim >= len(input_shape)):
        pytest.skip()

    if (input_shape in [(2, 32, 64, 64), (3, 22, 37, 41)] and dim in [0, -4, 1, -3]) or (
        input_shape in [(2, 32, 64), (3, 22, 37)] and dim in [0, -3]
    ):
        pytest.xfail("TTNN Issue: Unsupported dim")

    # TTNN Max issues:
    #   Unsupported dim - https://github.com/tenstorrent/tt-metal/issues/13186
    #   Shape mismatch along the H and W dimension - https://github.com/tenstorrent/tt-metal/issues/13189
    #   Tensor rank is not 4 - https://github.com/tenstorrent/tt-metal/issues/13190

    class ReduceMax(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.max(a, dim=dim, keepdim=True)[0]

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceMax()
    framework_model.eval()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.xfail(reason="Found Unsupported operations while lowering from TTForge to TTIR in forward graph")
@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, groups, bias, dilation, padding_mode, input_shape",
    [
        (16, 33, (3, 3), 2, 0, 1, True, 1, "zeros", (16, 50, 100)),
        (16, 32, (3, 5), 2, 1, 1, True, 1, "zeros", (16, 50, 100)),
        (16, 16, (3, 3), 1, 1, 16, True, 1, "zeros", (16, 50, 100)),
        (16, 33, (3, 3), 2, 0, 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 33, (3, 3), 2, 0, 1, False, 1, "zeros", (20, 16, 50, 100)),
        (16, 33, (3, 5), 2, 0, 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 16, (5, 5), 1, 2, 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 32, (3, 5), 2, 1, 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 32, (3, 3), 4, 1, 1, False, 1, "zeros", (20, 16, 50, 100)),
        (16, 16, (3, 3), 2, 2, 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 16, (3, 3), 1, 1, 16, True, 1, "zeros", (20, 16, 50, 100)),
    ],
)
def test_convtranspose2d(
    in_channels, out_channels, kernel_size, stride, padding, groups, bias, dilation, padding_mode, input_shape
):
    inputs = [torch.randn(*input_shape)]

    framework_model = torch.nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        padding_mode=padding_mode,
    )
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.xfail(
    reason="Unable to reshape a tensor in TILE_LAYOUT to non-tile height and width! Please convert the tensor to ROW_MAJOR_LAYOUT first"
)
@pytest.mark.push
def test_avg_pool2d():
    class AvgPool2d(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(
                x, kernel_size=[7, 7], stride=[7, 7], padding=(0, 0), ceil_mode=False, count_include_pad=True
            )

    inputs = [torch.rand(1, 2048, 7, 7)]

    framework_model = AvgPool2d()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize("shape", [(1, 3, 224, 224)])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.xfail(reason="RuntimeError: Tensor 1 - data type mismatch: expected BFloat16, got Float32")
@pytest.mark.push
def test_avgpool2d_decompose_to_conv2d(shape, padding):
    class AvgPool2d(nn.Module):
        def __init__(self, padding):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=[7, 7], stride=[7, 7], padding=padding)

        def forward(self, x):
            return self.pool(x)

    inputs = [torch.rand(shape).to(torch.bfloat16)]

    framework_model = AvgPool2d(padding=padding)
    framework_model = framework_model.to(torch.bfloat16)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize("shape", [(1, 3, 32, 32)])
@pytest.mark.parametrize(
    "padding",
    [
        pytest.param(
            (1, 1, 1, 1),
            marks=pytest.mark.xfail(reason="'ttnn.conv2d' op Bias must only have data on the final dimenstion"),
        ),
        pytest.param(
            (1, 1, 2, 2),
            marks=pytest.mark.xfail(reason="'ttnn.conv2d' op Bias must only have data on the final dimenstion"),
        ),
        pytest.param(
            (1, 2, 1, 2),
            marks=pytest.mark.xfail(
                reason="TTNN only supports padding height/width attributes. Thus, padding_top "
                "must equal padding_bottom for the op to execute as expected."
            ),
        ),
    ],
)
@pytest.mark.push
def test_conv2d_with_padding(shape, padding):
    class PaddingAndConv2d(nn.Module):
        def __init__(self, padding):
            super().__init__()
            self.padding = padding
            self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0)

        def forward(self, x):
            x = nn.functional.pad(x, self.padding, mode="constant", value=0)
            return self.conv(x)

    pad_top, pad_bottom, pad_left, pad_right = padding
    if pad_top != pad_bottom or pad_left != pad_right:
        pytest.xfail(
            "TTNN only supports padding height/width attributes. Thus, padding_top "
            "must equal padding_bottom for the op to execute as expected."
        )

    framework_model = PaddingAndConv2d(padding=padding)

    inputs = [torch.rand(shape)]
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.push
@pytest.mark.xfail(reason="Tensor rank is greater than 4")
def test_reshape_pytorch():
    class ReshapeTest(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp_1, inp_2):
            inp = inp_1 + inp_2
            inp_res = inp.reshape(1, 2, 2, 7, 7, 384)
            inp_res = inp_res.transpose(-4, -3)
            inp_res = inp_res.reshape(-1, 384)
            return inp_res

    inputs = [torch.rand(4, 49, 384), torch.rand(4, 49, 384)]
    framework_model = ReshapeTest()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.push
@pytest.mark.xfail(reason="Tensor rank is greater than 4")
def test_broadcast_pytorch():
    class BroadcastTest(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp_1):
            inp_1 = inp_1.transpose(-3, -2)
            inp_1_1 = inp_1[:1]
            inp_1_1 = inp_1_1.squeeze(0)
            return inp_1_1

    inputs = [torch.rand(3, 64, 49, 3, 32)]
    framework_model = BroadcastTest()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.xfail(
    reason="Unable to reshape a tensor in TILE_LAYOUT to non-tile height and width! Please convert the tensor to ROW_MAJOR_LAYOUT first"
)
@pytest.mark.parametrize(
    "params",
    [
        ([(1, 256, 24, 24), (1, 256, 24, 24)], -4),
        ([(5, 64, 128, 128), (5, 64, 128, 128)], -3),
        ([(1, 30, 30, 16), (1, 30, 30, 16)], -2),
        ([(1, 30, 30, 16), (1, 30, 30, 16)], 3),
        ([(5, 64, 128, 128), (5, 64, 128, 128)], -1),
        ([(1, 256, 24, 24), (1, 256, 24, 24)], 4),
        ([(1, 256, 24, 24), (1, 256, 24, 24)], 2),
        ([(5, 64, 128, 128), (5, 64, 128, 128)], 1),
        ([(1, 30, 30, 16), (1, 30, 30, 16)], 0),
    ],
)
def test_stack(params):
    class Stack(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, *tensors):
            return torch.stack(tensors, dim=self.dim)

    input_shapes, dim = params
    inputs = [torch.rand(shape) for shape in input_shapes]

    framework_model = Stack(dim)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name="stack_sanity")
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out


@pytest.mark.push
def test_remainder():
    class Remainder(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a % b

    inputs = [torch.rand(2, 32, 32), torch.rand(2, 32, 32)]

    framework_model = Remainder()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
