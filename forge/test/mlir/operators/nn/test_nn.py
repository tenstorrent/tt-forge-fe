# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn
import torch.nn.functional as F

import forge
from forge.verify.verify import verify


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
@pytest.mark.push
def test_conv2d_reflect_padding_mode(
    forge_property_recorder, input_shape, in_channels, out_channels, kernel_size, padding_value
):
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

    inputs = [torch.rand(input_shape)]

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
@pytest.mark.xfail(
    reason="permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 5 is not equal to len(dims) = 4. Tracking Issue: https://github.com/tenstorrent/tt-forge-fe/issues/1422"
)
@pytest.mark.push
def test_avgpool3d(forge_property_recorder, shape, kernel_size, stride):
    class AvgPool3D(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return nn.functional.avg_pool3d(x, kernel_size=kernel_size, stride=stride)

    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.compile_depth = (
        forge.CompileDepth.SPLIT_GRAPH
    )  # Due to #https://github.com/tenstorrent/tt-mlir/issues/1343
    inputs = [torch.rand(shape)]

    framework_model = AvgPool3D()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, forge_property_handler=forge_property_recorder
    )

    if compiler_cfg.compile_depth == forge.CompileDepth.FULL:
        verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride_size, padding, ceil_mode",
    [
        pytest.param(
            (1, 96, 54, 54),
            3,
            2,
            0,
            True,
        ),
        pytest.param(
            (1, 64, 55, 54),
            3,
            2,
            0,
            True,
        ),
        pytest.param(
            (1, 128, 26, 26),
            3,
            2,
            0,
            True,
        ),
        pytest.param(
            (1, 256, 26, 26),
            3,
            2,
            0,
            True,
        ),
        pytest.param(
            (1, 96, 54, 54),
            3,
            2,
            0,
            False,
        ),
        pytest.param(
            (1, 64, 55, 54),
            3,
            2,
            0,
            False,
        ),
        pytest.param(
            (1, 128, 26, 26),
            3,
            2,
            0,
            False,
        ),
        pytest.param(
            (1, 256, 26, 26),
            3,
            2,
            0,
            False,
        ),
        pytest.param(
            (1, 3, 32, 32),
            3,
            3,
            (1, 1, 1, 1),
            False,
            marks=pytest.mark.xfail(
                reason="Runtime Error  : Shard page size must currently have L1 aligned page size."
            ),
        ),
        pytest.param(
            (1, 3, 32, 32),
            3,
            3,
            (1, 1, 2, 2),
            False,
            marks=pytest.mark.xfail(
                reason="Runtime Error  : Shard page size must currently have L1 aligned page size."
            ),
        ),
    ],
)
@pytest.mark.push
def test_maxpool2d(forge_property_recorder, input_shape, kernel_size, stride_size, padding, ceil_mode):
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
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "shape, mode",
    [
        pytest.param((1, 2048, 7, 7), "nearest"),
        pytest.param(
            (1, 2048, 7, 7), "bilinear", marks=pytest.mark.xfail(reason="Runtime Error TTNN: info: Unsupported mode ")
        ),
    ],
)
@pytest.mark.push
def test_interpolate(forge_property_recorder, shape, mode):
    class Interpolate(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return nn.functional.interpolate(x, scale_factor=2, mode=mode)

    inputs = [torch.rand(shape)]

    framework_model = Interpolate()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
def test_batchnorm2d(forge_property_recorder, batch_size, num_channels, height, width):

    if batch_size != 1:
        pytest.xfail("Batch size is not 1")

    inputs = [torch.rand(batch_size, num_channels, height, width)]

    framework_model = nn.BatchNorm2d(num_features=num_channels)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.skip(reason="This is not ready yet")
@pytest.mark.push
def test_linear(forge_property_recorder):
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(20, 30, bias=True)

        def forward(self, a):
            return self.l1(a)

    inputs = [torch.rand(1, 128, 20)]

    framework_model = Linear()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_softmax(forge_property_recorder):
    class Softmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, a):
            return self.softmax(a)

    inputs = [torch.rand(1, 128)]

    framework_model = Softmax()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 128), 1),
        ((4, 32), 1),
        ((2, 3, 5), 2),
        ((2, 3, 5), -1),
        ((10,), 0),
    ],
)
def test_log_softmax(forge_property_recorder, input_shape, dim):
    class LogSoftmax(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.log_softmax = nn.LogSoftmax(dim=dim)

        def forward(self, a):
            return self.log_softmax(a)

    inputs = [torch.rand(*input_shape)]

    framework_model = LogSoftmax(dim)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


# @pytest.mark.parametrize("vocab_size", [2048, 16384, 32000])
# @pytest.mark.parametrize("token_num", [1, 7, 32])
# @pytest.mark.parametrize("embedding_dim", [128, 512, 3200])
@pytest.mark.parametrize("vocab_size", [32000])
@pytest.mark.parametrize("token_num", [12])
@pytest.mark.parametrize("embedding_dim", [3200])
@pytest.mark.push
def test_embedding(forge_property_recorder, vocab_size, token_num, embedding_dim):
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.enable_tvm_cpu_fallback = False

    class Embedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            return self.embedding(x)

    inputs = [
        torch.randint(0, vocab_size, (1, token_num)),
    ]

    framework_model = Embedding()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, groups, bias, dilation, padding_mode, input_shape",
    [
        pytest.param(
            16,
            33,
            (3, 3),
            2,
            0,
            1,
            True,
            1,
            "zeros",
            (16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 2676352 B L1 buffer"),
        ),
        pytest.param(
            16,
            32,
            (3, 5),
            2,
            1,
            1,
            True,
            1,
            "zeros",
            (16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 2650240 B L1 buffer"),
        ),
        pytest.param(
            16,
            16,
            (3, 3),
            1,
            1,
            16,
            True,
            1,
            "zeros",
            (16, 50, 100),
            marks=pytest.mark.xfail(
                reason="Circular buffers clash with L1 buffers: static circular buffer region ends at 745248"
            ),
        ),
        pytest.param(
            16,
            33,
            (3, 3),
            1,
            (0, 0),
            1,
            True,
            1,
            "zeros",
            (16, 50, 100),
            marks=pytest.mark.xfail(
                reason="Circular buffers clash with L1 buffers: static circular buffer region ends at 745248"
            ),
        ),
        pytest.param(
            16,
            33,
            (3, 3),
            1,
            (1, 0),
            1,
            True,
            1,
            "zeros",
            (16, 50, 100),
            marks=pytest.mark.xfail(
                reason="Statically allocated circular buffers in program 33 clash with L1 buffers on core"
            ),
        ),
        pytest.param(
            16,
            33,
            (3, 3),
            1,
            (0, 1),
            1,
            True,
            1,
            "zeros",
            (16, 50, 100),
            marks=pytest.mark.xfail(
                reason="Statically allocated circular buffers in program 39 clash with L1 buffers on core"
            ),
        ),
        pytest.param(
            16,
            33,
            (3, 3),
            2,
            0,
            1,
            True,
            1,
            "zeros",
            (20, 16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 12800000 B L1 buffer"),
        ),
        pytest.param(
            16,
            33,
            (3, 3),
            2,
            0,
            1,
            False,
            1,
            "zeros",
            (20, 16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 12800000 B L1 buffer"),
        ),
        pytest.param(
            16,
            33,
            (3, 5),
            2,
            0,
            1,
            True,
            1,
            "zeros",
            (20, 16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 12800000 B L1 buffer"),
        ),
        pytest.param(
            16,
            16,
            (5, 5),
            1,
            2,
            1,
            True,
            1,
            "zeros",
            (20, 16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 12800000 B L1 buffer"),
        ),
        pytest.param(
            16,
            32,
            (3, 5),
            2,
            1,
            1,
            True,
            1,
            "zeros",
            (20, 16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 12800000 B L1 buffer"),
        ),
        pytest.param(
            16,
            32,
            (3, 3),
            4,
            (1, 2),
            1,
            False,
            1,
            "zeros",
            (20, 16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 12800000 B L1 buffer"),
        ),
        pytest.param(
            16,
            16,
            (3, 3),
            2,
            (2, 3),
            1,
            True,
            1,
            "zeros",
            (20, 16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 12800000 B L1 buffer"),
        ),
        pytest.param(
            16,
            16,
            (3, 3),
            1,
            (3, 3),
            16,
            True,
            1,
            "zeros",
            (20, 16, 50, 100),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 12800000 B L1 buffer"),
        ),
        pytest.param(
            64,
            128,
            (7, 7),
            4,
            (3, 5),
            1,
            False,
            1,
            "zeros",
            (16, 64, 80, 80),
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 26214400 B L1 buffer"),
        ),
        pytest.param(
            32,
            32,
            (1, 1),
            1,
            (5, 6),
            1,
            False,
            1,
            "zeros",
            (10, 32, 20, 20),
            marks=pytest.mark.xfail(reason="Tensor mismatch. PCC = 0.0014"),
        ),
    ],
)
def test_convtranspose2d(
    forge_property_recorder,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups,
    bias,
    dilation,
    padding_mode,
    input_shape,
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

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.push
def test_avg_pool2d(forge_property_recorder):
    class AvgPool2d(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(
                x, kernel_size=[7, 7], stride=[7, 7], padding=(0, 0), ceil_mode=False, count_include_pad=True
            )

    inputs = [torch.rand(1, 2048, 7, 7)]

    framework_model = AvgPool2d()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("shape", [(1, 3, 224, 224)])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.xfail(reason="RuntimeError: Tensor 1 - data type mismatch: expected BFloat16, got Float32")
@pytest.mark.push
def test_avgpool2d_decompose_to_conv2d(forge_property_recorder, shape, padding):
    class AvgPool2d(nn.Module):
        def __init__(self, padding):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=[7, 7], stride=[7, 7], padding=padding)

        def forward(self, x):
            return self.pool(x)

    inputs = [torch.rand(shape).to(torch.bfloat16)]

    framework_model = AvgPool2d(padding=padding)
    framework_model = framework_model.to(torch.bfloat16)

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("shape", [(1, 3, 32, 32)])
@pytest.mark.parametrize(
    "padding",
    [
        pytest.param((1, 1, 1, 1)),
        pytest.param((1, 1, 2, 2)),
        pytest.param(
            (1, 2, 1, 2),
            marks=pytest.mark.xfail(
                reason="RuntimeError: ttnn.pad: on device tile padding does not support front padding"
            ),
        ),
    ],
)
@pytest.mark.push
def test_conv2d_with_padding(forge_property_recorder, shape, padding):
    class PaddingAndConv2d(nn.Module):
        def __init__(self, padding):
            super().__init__()
            self.padding = padding
            self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0)

        def forward(self, x):
            x = nn.functional.pad(x, self.padding, mode="constant", value=0)
            return self.conv(x)

    inputs = [torch.rand(shape)]

    framework_model = PaddingAndConv2d(padding=padding)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - adv_index"
)
@pytest.mark.parametrize(
    "img, grid",
    [
        ((1, 2, 4, 4), (1, 6, 2, 2)),
        ((1, 32, 50, 50), (1, 2500, 4, 2)),
        ((1, 3, 8, 8), (1, 3, 3, 2)),
        ((1, 3, 16, 16), (1, 8, 8, 2)),
        ((5, 2, 10, 10), (5, 12, 3, 2)),
        ((3, 8, 32, 32), (3, 25, 4, 2)),
    ],
)
@pytest.mark.parametrize("align_corners", [True, False])
def test_grid_sample(forge_property_recorder, img, grid, align_corners, test_device):
    class GridSampleModule(nn.Module):
        def __init__(self, interpolation="bilinear", align_corners=align_corners):
            super(GridSampleModule, self).__init__()
            self.interpolation = interpolation
            self.align_corners = align_corners

        def forward(self, img, grid):
            output = F.grid_sample(img, grid, mode=self.interpolation, align_corners=align_corners)
            return output

    # TO-DO: Support for nearest interpolation mode is yet to be added
    model = GridSampleModule(interpolation="bilinear", align_corners=align_corners)
    model.eval()
    img = torch.randn(img)
    grid = torch.randn(grid)
    output = model(img, grid)
    compiled_model = forge.compile(
        model, sample_inputs=[img, grid], module_name="grid_sample", forge_property_handler=forge_property_recorder
    )
