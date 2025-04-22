# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn
import torch.nn.functional as F

import tensorflow as tf
import onnx
import onnxruntime as ort
import onnx.helper
import onnx.numpy_helper

import forge
from forge.verify.verify import verify
from forge.config import CompilerConfig
from forge._C import DataFormat


# PyTorch test remains the same
@pytest.mark.parametrize("shape", [(1, 3, 8, 8)])
@pytest.mark.push
def test_conv2d_bnorm_bfloat16_pytorch(forge_property_recorder, shape):
    class TinyBNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(16, eps=1e-5)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    inputs = [torch.rand(shape).to(torch.bfloat16)]
    framework_model = TinyBNNet().to(torch.bfloat16)

    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, groups, bias, dilation, padding_mode, input_shape",
    [
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
            marks=pytest.mark.xfail(reason="Out of Memory: Not enough space to allocate 13107200 B L1 buffer"),
        ),
    ],
)
@pytest.mark.push
def test_convtranspose2d_bfloat16_pytorch(
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
    inputs = [torch.randn(*input_shape).to(torch.bfloat16)]

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
    ).to(torch.bfloat16)

    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder, compiler_cfg=compiler_cfg
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("shape", [(1, 3, 32, 32)])
@pytest.mark.parametrize(
    "padding",
    [
        pytest.param((1, 1, 1, 1)),
    ],
)
@pytest.mark.push
def test_conv2d_and_matmul_bfloat16_pytorch(forge_property_recorder, shape, padding):
    class PaddingAndConv2d(nn.Module):
        def __init__(self, padding):
            super().__init__()
            self.padding = padding
            self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0)

        def forward(self, x):
            k = nn.functional.pad(x, self.padding, mode="constant", value=0)
            y = self.conv(k)
            return y @ x
            # return self.conv(x)

    pad_top, pad_bottom, pad_left, pad_right = padding
    if pad_top != pad_bottom or pad_left != pad_right:
        pytest.xfail(
            "TTNN only supports padding height/width attributes. Thus, padding_top "
            "must equal padding_bottom for the op to execute as expected."
        )

    inputs = [torch.rand(shape).to(torch.bfloat16)]

    framework_model = PaddingAndConv2d(padding=padding).to(torch.bfloat16)
    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder, compiler_cfg=compiler_cfg
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
