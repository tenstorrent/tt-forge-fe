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


@pytest.mark.xfail(reason="error: 'ttnn.conv2d' op Bias must only have data on the final dimenstion")
@pytest.mark.parametrize(
    "input_shape, in_channels, out_channels, kernel_size, padding_value",
    [
        ((1, 512, 6, 20), 512, 256, 3, 1),
        ((1, 128, 32, 32), 128, 64, 5, 2),
        ((1, 64, 64, 64), 64, 128, 3, 1),
        ((1, 32, 128, 128), 32, 64, 7, 3),
        ((1, 256, 16, 16), 256, 128, 5, 2),
    ],
)
def test_conv2d_reflect_padding_mode(input_shape, in_channels, out_channels, kernel_size, padding_value):
    class Conv2dReflectPad(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding_value):
            super().__init__()
            self.pad = nn.ReflectionPad2d(padding_value)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

        def forward(self, input):
            out = self.pad(input)
            out = self.conv(out)
            return out

    framework_model = Conv2dReflectPad(in_channels, out_channels, kernel_size, padding_value)
    framework_model.eval()

    inputs = torch.rand(input_shape)

    compiled_model = forge.compile(framework_model, sample_inputs=[inputs])

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(reason="RuntimeError: Input must be UINT32 or BFLOAT16")
@pytest.mark.parametrize(
    "input_shape, sequence_lengths",
    [
        ((1, 32, 2), [5]),
        ((1, 64, 4), [55]),
        ((1, 16, 8), [14]),
        ((1, 164, 14), [22]),
        ((1, 80, 7), [79]),
        ((1, 43, 25), [34]),
    ],
)
def test_multi_indexing(input_shape, sequence_lengths):
    class Multi_Indexing(torch.nn.Module):
        def __init__(self, sequence_lengths):
            super().__init__()
            self.sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.int64)

        def forward(self, logits):
            pooled_logits = logits[torch.arange(1), self.sequence_lengths]
            return pooled_logits

    inputs = [torch.randn(input_shape)]
    framework_model = Multi_Indexing(sequence_lengths)
    framework_model.eval()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "shape,dim,index",
    [
        ((3, 23, 73, 164), 1, 21),
        ((8, 66, 713, 54), 2, -403),
        ((12, 86, 273, 34), 3, 30),
        ((5, 115, 75, 64), -3, -21),
        ((2, 7, 213, 64), -2, -103),
        ((6, 99, 12, 64), -1, 36),
        ((1, 6, 73, 64), 2, -2),
        ((1, 6, 73, 64), -2, -1),
        ((3, 27, 94), 0, 2),
        ((5, 100, 64), 1, 50),
        ((7, 82, 16), 2, -5),
        ((10, 53, 23), -1, 15),
        ((8, 32, 12), -2, -20),
        ((18, 31, 22), -3, -12),
    ],
)
@pytest.mark.push
def test_index(shape, dim, index):
    class Index(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, x):

            if dim == 3 or dim == -1:
                return x[..., [self.index]]

            elif dim == 2 or dim == -2:
                return x[..., [self.index], :]

            elif dim == 1 or dim == -3:
                return x[:, [self.index], ...]

            else:
                return x[[self.index], ...]

    inputs = [torch.rand(shape)]

    framework_model = Index(index)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    if compiler_cfg.compile_depth == forge.CompileDepth.FULL:
        verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    "batch_size, num_channels, height, width",
    [
        (1, 32, 56, 56),
    ],
)
@pytest.mark.push
def test_layernorm(batch_size, num_channels, height, width):

    # framework_model = nn.LayerNorm((num_channels, height, width)) # Support only normalization over last one dimension
    framework_model = nn.LayerNorm((width))

    inputs = [torch.rand(batch_size, num_channels, height, width)]

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

    inputs = [condition, input, other]

    framework_model = Where()
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

    inputs = [x, y]

    framework_model = Maximum()
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

    if batch_size != 1:
        pytest.xfail("Batch size is not 1")

    inputs = [torch.rand(batch_size, num_channels, height, width)]

    framework_model = nn.BatchNorm2d(num_features=num_channels)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_add():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    inputs = [torch.rand(2, 32, 32), torch.rand(2, 32, 32)]

    framework_model = Add()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        ((64,), 0, True),
        ((64,), -1, True),
        ((4, 64), 0, True),
        ((32, 32), -2, True),
        pytest.param((2, 32, 32), 0, True, marks=pytest.mark.xfail(reason="tt:exception Unsupported dim")),
        ((1, 64, 32), 2, True),
        ((4, 32, 64), -2, True),
        ((4, 128, 128, 128), 0, True),
        ((1, 128, 128, 128), 2, True),
        ((1, 128, 128, 128), -3, True),
        ((4, 128, 128, 128), -4, True),
        pytest.param(
            (64,),
            0,
            False,
            marks=pytest.mark.xfail(reason="'ttir.squeeze' op Output tensor must have at least one dimension."),
        ),
        pytest.param(
            (64,),
            -1,
            False,
            marks=pytest.mark.xfail(reason="'ttir.squeeze' op Output tensor must have at least one dimension."),
        ),
        pytest.param(
            (4, 64),
            0,
            False,
        ),
        pytest.param(
            (32, 32),
            -2,
            False,
        ),
        pytest.param((2, 32, 32), 0, False, marks=pytest.mark.xfail(reason="tt:exception Unsupported dim")),
        ((1, 64, 32), 2, False),
        ((4, 32, 64), -2, False),
        ((4, 128, 128, 128), 0, False),
        (
            (1, 128, 128, 128),
            2,
            False,
        ),
        ((1, 128, 128, 128), -3, False),
        ((4, 128, 128, 128), -4, False),
    ],
)
@pytest.mark.push
def test_reduce_sum(input_shape, dim, keepdim):
    class ReduceSum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.sum(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceSum()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    # Skipping PCC check due to inconsistencies between Framework and Compiled model
    #
    # co_out = [co.to("cpu") for co in co_out]
    # assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        ((64,), 0, True),
        ((64,), -1, True),
        ((4, 64), 0, True),
        ((32, 32), -2, True),
        pytest.param((2, 32, 32), 0, True, marks=pytest.mark.xfail(reason="tt:exception Unsupported dim")),
        ((1, 64, 32), 2, True),
        ((4, 32, 64), -2, True),
        ((4, 128, 128, 128), 0, True),
        ((1, 128, 128, 128), 2, True),
        ((1, 128, 128, 128), -3, True),
        ((4, 128, 128, 128), -4, True),
        pytest.param(
            (64,),
            0,
            False,
            marks=pytest.mark.xfail(reason="'ttir.squeeze' op Output tensor must have at least one dimension."),
        ),
        pytest.param(
            (64,),
            -1,
            False,
            marks=pytest.mark.xfail(reason="'ttir.squeeze' op Output tensor must have at least one dimension."),
        ),
        pytest.param(
            (4, 64),
            0,
            False,
        ),
        pytest.param(
            (32, 32),
            -2,
            False,
        ),
        pytest.param((2, 32, 32), 0, False, marks=pytest.mark.xfail(reason="tt:exception Unsupported dim")),
        ((1, 64, 32), 2, False),
        ((4, 32, 64), -2, False),
        ((4, 128, 128, 128), 0, False),
        (
            (1, 128, 128, 128),
            2,
            False,
        ),
        ((1, 128, 128, 128), -3, False),
        ((4, 128, 128, 128), -4, False),
    ],
)
@pytest.mark.push
def test_reduce_mean(input_shape, dim, keepdim):
    class ReduceMean(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.mean(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceMean()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    # Skipping PCC check due to inconsistencies between Framework and Compiled model
    #
    # co_out = [co.to("cpu") for co in co_out]
    # assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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

    # Skipping PCC check due to inconsistencies between Framework and Compiled model
    #
    # co_out = [co.to("cpu") for co in co_out]
    # assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


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

    framework_model = ForgeIndexing(dim, start, stop, stride)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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

    framework_model = ForgeAdvIndex("ForgeAdvIndex")

    # Sample Inputs
    pt_input_tensor = torch.rand(input_tensor_shape).to(torch.float32)
    pt_indices = torch.randint(input_tensor_shape[0], indices_shape).to(torch.int32)
    inputs = to_forge_tensors([pt_input_tensor, pt_indices])

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        ((64,), 0, True),
        ((64,), -1, True),
        ((4, 64), 0, True),
        ((32, 32), -2, True),
        pytest.param((2, 32, 32), 0, True, marks=pytest.mark.xfail(reason="tt:exception Unsupported dim")),
        ((1, 64, 32), 2, True),
        ((4, 32, 64), -2, True),
        ((4, 128, 128, 128), 0, True),
        ((1, 128, 128, 128), 2, True),
        ((1, 128, 128, 128), -3, True),
        ((4, 128, 128, 128), -4, True),
        pytest.param(
            (64,),
            0,
            False,
            marks=pytest.mark.xfail(reason="[mlir::AffineMap collapsedLinearAffineMap] Assertion `end > 0' failed."),
        ),
        pytest.param(
            (64,),
            -1,
            False,
            marks=pytest.mark.xfail(reason="[mlir::AffineMap collapsedLinearAffineMap] Assertion `end > 0' failed."),
        ),
        pytest.param(
            (4, 64),
            0,
            False,
            marks=pytest.mark.xfail(reason="keepdim=False is not supported"),
        ),
        pytest.param(
            (32, 32),
            -2,
            False,
            marks=pytest.mark.xfail(reason="keepdim=False is not supported"),
        ),
        pytest.param((2, 32, 32), 0, False, marks=pytest.mark.xfail(reason="keepdim=False is not supported")),
        pytest.param((1, 64, 32), 2, False, marks=pytest.mark.xfail(reason="keepdim=False is not supported")),
        pytest.param((4, 32, 64), -2, False, marks=pytest.mark.xfail(reason="keepdim=False is not supported")),
        pytest.param((4, 128, 128, 128), 0, False, marks=pytest.mark.xfail(reason="keepdim=False is not supported")),
        pytest.param(
            (1, 128, 128, 128),
            2,
            False,
            marks=pytest.mark.xfail(reason="keepdim=False is not supported"),
        ),
        pytest.param((1, 128, 128, 128), -3, False, marks=pytest.mark.xfail(reason="keepdim=False is not supported")),
        pytest.param((4, 128, 128, 128), -4, False, marks=pytest.mark.xfail(reason="keepdim=False is not supported")),
    ],
)
@pytest.mark.push
def test_reduce_max(input_shape, dim, keepdim):
    input = (input_shape, dim, keepdim)
    if input in [((64,), 0, False), ((64,), -1, False)]:
        pytest.xfail(reason="[mlir::AffineMap collapsedLinearAffineMap] Assertion `end > 0' failed.")

    # TTNN Max issues:
    #   Unsupported dim - https://github.com/tenstorrent/tt-metal/issues/13186
    #   Shape mismatch along the H and W dimension - https://github.com/tenstorrent/tt-metal/issues/13189
    #   Tensor rank is not 4 - https://github.com/tenstorrent/tt-metal/issues/13190

    class ReduceMax(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.max(a, dim=dim, keepdim=keepdim)[0]

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceMax()
    framework_model.eval()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]

    # Skipping PCC check due to inconsistencies between Framework and Compiled model
    #
    # co_out = [co.to("cpu") for co in co_out]
    # assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.xfail(reason="Found Unsupported operations while lowering from TTForge to TTIR in forward graph")
@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, groups, bias, dilation, padding_mode, input_shape",
    [
        (16, 33, (3, 3), 2, 0, 1, True, 1, "zeros", (16, 50, 100)),
        (16, 32, (3, 5), 2, 1, 1, True, 1, "zeros", (16, 50, 100)),
        (16, 16, (3, 3), 1, 1, 16, True, 1, "zeros", (16, 50, 100)),
        (16, 33, (3, 3), 1, (0, 0), 1, True, 1, "zeros", (16, 50, 100)),
        (16, 33, (3, 3), 1, (1, 0), 1, True, 1, "zeros", (16, 50, 100)),
        (16, 33, (3, 3), 1, (0, 1), 1, True, 1, "zeros", (16, 50, 100)),
        (16, 33, (3, 3), 2, 0, 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 33, (3, 3), 2, 0, 1, False, 1, "zeros", (20, 16, 50, 100)),
        (16, 33, (3, 5), 2, 0, 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 16, (5, 5), 1, 2, 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 32, (3, 5), 2, 1, 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 32, (3, 3), 4, (1, 2), 1, False, 1, "zeros", (20, 16, 50, 100)),
        (16, 16, (3, 3), 2, (2, 3), 1, True, 1, "zeros", (20, 16, 50, 100)),
        (16, 16, (3, 3), 1, (3, 3), 16, True, 1, "zeros", (20, 16, 50, 100)),
        (64, 128, (7, 7), 4, (3, 5), 1, False, 1, "zeros", (16, 64, 80, 80)),
        (32, 32, (1, 1), 1, (5, 6), 1, False, 1, "zeros", (10, 32, 20, 20)),
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

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(
    reason="RuntimeError: TT_FATAL @ /tt-metal/src/tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp:474: new_volume == old_volume. Invalid arguments to reshape. Tracking on: https://github.com/tenstorrent/tt-mlir/issues/1574"
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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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

    inputs = [torch.rand(shape)]

    framework_model = PaddingAndConv2d(padding=padding)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - repeat"
)
@pytest.mark.push
def test_repeat():
    class Repeat(nn.Module):
        def __init__(self, repeats):
            super().__init__()
            self.repeats = repeats

        def forward(self, x):
            return x.repeat(*self.repeats)

    inputs = [torch.rand(1, 2, 1, 4, 4)]

    framework_model = Repeat(repeats=(1, 1, 4, 1, 1))
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - repeat_interleave"
)
@pytest.mark.push
def test_expand():
    class Expand(nn.Module):
        def __init__(self, expand_shape):
            super().__init__()
            self.expand_shape = expand_shape

        def forward(self, x):
            return x.expand(*self.expand_shape)

    inputs = [torch.rand(1, 2, 1, 4, 4)]

    framework_model = Expand(expand_shape=(1, 2, 4, 4, 4))
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - repeat_interleave"
)
@pytest.mark.push
def test_repeat_interleave():
    class RepeatInterleave(nn.Module):
        def __init__(self, repeats, dim):
            super().__init__()
            self.repeats = repeats
            self.dim = dim

        def forward(self, x):
            return x.repeat_interleave(self.repeats, dim=self.dim)

    inputs = [torch.rand(1, 2, 1, 4, 4)]

    framework_model = RepeatInterleave(repeats=4, dim=2)
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


@pytest.mark.parametrize("shape", [(1, 32, 64, 64), (32, 64, 64), (64, 64)])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("begin", [0, 16])
@pytest.mark.parametrize("length", [4, 16])
@pytest.mark.parametrize("stride", [16, 32])
def test_select(shape, dim, begin, length, stride):
    if stride <= begin + length:
        pytest.skip("Skipping since stride <= begin + length")

    class Select(forge.ForgeModule):
        def __init__(self):
            super().__init__("Select")

        def forward(self, x):
            x = forge.op.Select("select_op", x, dim, [begin, length], stride)
            return x

    inputs = to_forge_tensors([torch.rand(*shape)])
    framework_model = Select()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
