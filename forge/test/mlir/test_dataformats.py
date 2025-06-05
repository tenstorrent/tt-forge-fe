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
from forge.verify.verify import verify, VerifyConfig
from forge.config import CompilerConfig
from forge._C import DataFormat


@pytest.mark.parametrize(
    "shape",
    [
        (2, 32, 32),
    ],
)
@pytest.mark.push
def test_add_bfloat16_pytorch(shape):
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    # Generate random tensors of the appropriate shape and dtype
    a = torch.rand(size=shape).to(torch.bfloat16)
    b = torch.rand(size=shape).to(torch.bfloat16)
    inputs = [a, b]

    framework_model = Add()
    framework_model = framework_model.to(torch.bfloat16)

    data_format_override = DataFormat.Float16_b

    compiler_cfg = CompilerConfig(default_df_override=data_format_override)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(
        inputs,
        framework_model,
        compiled_model,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (2, 32, 32),
    ],
)
@pytest.mark.push
def test_add_constant_bfloat16_pytorch(shape):
    class Add(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("constant_b", torch.rand(size=shape))

        def forward(self, a):
            return a + self.constant_b

    # Generate random tensors of the appropriate shape and dtype
    a = torch.rand(size=shape).to(torch.bfloat16)
    inputs = [
        a,
    ]

    framework_model = Add()
    framework_model = framework_model.to(torch.bfloat16)

    data_format_override = DataFormat.Float16_b

    compiler_cfg = CompilerConfig(default_df_override=data_format_override)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(
        inputs,
        framework_model,
        compiled_model,
    )


@pytest.mark.parametrize("shape", [(1, 3, 8, 8)])
@pytest.mark.push
def test_conv2d_bnorm_bfloat16_pytorch(shape):
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
        compiler_cfg=compiler_cfg,
    )

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("shape", [(1, 3, 32, 32)])
@pytest.mark.parametrize(
    "padding",
    [
        pytest.param((1, 1, 1, 1)),
    ],
)
@pytest.mark.push
def test_conv2d_and_matmul_bfloat16_pytorch(shape, padding):
    class PaddingAndConv2d(nn.Module):
        def __init__(self, padding):
            super().__init__()
            self.padding = padding
            self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0)

        def forward(self, x):
            k = nn.functional.pad(x, self.padding, mode="constant", value=0)
            y = self.conv(k)
            return y @ x

    pad_top, pad_bottom, pad_left, pad_right = padding
    if pad_top != pad_bottom or pad_left != pad_right:
        pytest.xfail(
            "TTNN only supports padding height/width attributes. Thus, padding_top "
            "must equal padding_bottom for the op to execute as expected."
        )

    inputs = [torch.rand(shape).to(torch.bfloat16)]

    framework_model = PaddingAndConv2d(padding=padding).to(torch.bfloat16)
    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("vocab_size", [32000])
@pytest.mark.parametrize("token_num", [12])
@pytest.mark.parametrize("embedding_dim", [3200])
@pytest.mark.push
def test_embedding(vocab_size, token_num, embedding_dim):
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    class Embedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            return self.embedding(x)

    inputs = [
        torch.randint(0, vocab_size, (1, token_num)),
    ]

    framework_model = Embedding().to(torch.bfloat16)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("vocab_size", [32000])
@pytest.mark.parametrize("token_num", [12])
@pytest.mark.parametrize("embedding_dim", [3200])
@pytest.mark.push
def test_embedding_const_eval_pass(vocab_size, token_num, embedding_dim):
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    class ModelConstEvalPass(nn.Module):
        def __init__(self):
            super().__init__()

            self.const = torch.randint(0, vocab_size, (1, token_num), dtype=torch.int32)
            self.register_buffer("constant", self.const)

            self.embedding = nn.Embedding(vocab_size, embedding_dim, dtype=torch.bfloat16)

        def forward(self, x):
            v1 = self.embedding(self.constant)
            v2 = self.embedding(x)
            add = torch.add(v1, v2)
            return add

    inputs = [
        torch.randint(0, vocab_size, (1, token_num)),
    ]

    framework_model = ModelConstEvalPass().to(torch.bfloat16)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model)


    # from forge.verify.value_checkers import AllCloseValueChecker
    
    # verify(inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AllCloseValueChecker()))
