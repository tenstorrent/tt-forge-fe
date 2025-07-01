# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
import math
import onnx


@pytest.mark.xfail(
    reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen, PCC got=0.4923030518607919"
)  # https://github.com/tenstorrent/tt-forge-fe/issues/1793
@pytest.mark.push
def test_inplace_updation():
    class Inplace_updation(nn.Module):
        def __init__(self):
            super().__init__()
            self.shift_size = 4
            self.window_size = 8

        def forward(self, x):

            img_mask = torch.zeros((1, 64, 64, 1), dtype=torch.float32)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            return img_mask + x

    inputs = [torch.randn((1, 64, 64, 1), dtype=torch.float32)]
    model = Inplace_updation()
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )
    verify(inputs, model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, clamp_min, clamp_max, dtype",
    [
        ((64, 3, 64, 64), None, math.log(100.0), torch.float32),
        ((11, 30, 11, 11), math.log(101.0), None, torch.float32),
        ((27, 16, 27, 27), -math.log(103.0), None, torch.float32),
        ((45, 2, 45, 45), None, 5.0, torch.float32),
        ((3, 21, 3, 3), None, math.log(50.0), torch.float32),
        ((12, 6, 12, 12), -103.0, None, torch.float32),
        pytest.param(
            (18, 11, 18, 18),
            -50,
            None,
            torch.int32,
        ),
        pytest.param(
            (8, 1, 8, 8),
            None,
            876,
            torch.int32,
        ),
    ],
)
@pytest.mark.push
def test_clamp(input_shape, clamp_min, clamp_max, dtype):
    class Clamp(nn.Module):
        def __init__(self):
            super().__init__()
            log_value = torch.log(10 * torch.ones(input_shape[1], 1, 1))
            self.logit_scale = nn.Parameter(log_value)

        def forward(self, attn):
            clamped = torch.clamp(self.logit_scale, min=clamp_min, max=clamp_max)
            logit_scale = clamped.exp()
            return attn * logit_scale

    model = Clamp()
    model.eval()

    if dtype == torch.float32:
        attn = torch.randn(input_shape)
    elif dtype == torch.int32:
        attn = torch.randint(low=torch.iinfo(dtype).min, high=torch.iinfo(dtype).max, size=input_shape, dtype=dtype)

    inputs = [attn]

    # Export to ONNX
    torch.onnx.export(model, (inputs[0],), "temp.onnx", opset_version=17)

    # Load and check ONNX
    onnx_model = onnx.load("temp.onnx")
    onnx.checker.check_model(onnx_model)

    compiled_model = forge.compile(onnx_model, inputs)
    verify(inputs, model, compiled_model)


@pytest.mark.push
def test_rotary_pos_emb():
    class RotaryPosEmb(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q, cos, sin):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

            rotary_dim = cos.shape[-1]
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]

            q_embed = torch.cat([q_pass, (q_rot * cos) + (self.rotate_half(q_rot) * sin)], dim=-1)
            return q_embed

        def rotate_half(self, x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

    framework_model = RotaryPosEmb()
    framework_model.eval()

    query = torch.rand((1, 32, 256, 96))
    cos_emb = torch.rand([1, 256, 96])
    sin_emb = torch.rand([1, 256, 96])
    inputs = [query, cos_emb, sin_emb]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
@pytest.mark.parametrize(
    "dim",
    [-4, -3, -2, -1, 0, 1, 2, 3],
)
def test_remove_concat_pass(dim):
    class SliceConcat(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, q):
            sl = [slice(None)] * len(q.shape)
            sl[self.dim] = slice(q.shape[self.dim], None)
            q_pass = q[tuple(sl)]
            q_embed = torch.cat([q_pass, (q * q)], dim=self.dim)
            return q_embed

    model = SliceConcat(dim)

    query = torch.rand((1, 32, 256, 96))
    inputs = [query]

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)
