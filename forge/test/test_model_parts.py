# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
from forge.verify import DepricatedVerifyConfig


@pytest.mark.xfail(
    reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen, PCC got=0.4923030518607919"
)  # https://github.com/tenstorrent/tt-forge-fe/issues/1793
@pytest.mark.push
def test_inplace_updation(forge_property_recorder):
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
        verify_cfg=DepricatedVerifyConfig(verify_forge_codegen_vs_framework=True),
        forge_property_handler=forge_property_recorder,
    )
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("inplace", [True, False])
def test_attn_mask(inplace):
    class Attn_mask_model(nn.Module):
        def __init__(self, inplace=True):
            super().__init__()
            self.inplace = inplace
            self.pad_H = 56
            self.window_size = [7, 7]
            self.pad_W = 56
            self.num_heads = 3
            self.num_windows = 64
            self.shift_size = [3, 3]

        def forward(self, x, attn):

            attn_mask = x.new_zeros((self.pad_H, self.pad_W))
            if self.inplace == True:
                h_slices = (
                    (0, -self.window_size[0]),
                    (-self.window_size[0], -self.shift_size[0]),
                    (-self.shift_size[0], None),
                )
                w_slices = (
                    (0, -self.window_size[1]),
                    (-self.window_size[1], -self.shift_size[1]),
                    (-self.shift_size[1], None),
                )
                count = 0
                for h in h_slices:
                    for w in w_slices:
                        attn_mask[h[0] : h[1], w[0] : w[1]] = count
                        count += 1

            attn_mask = attn_mask.view(
                self.pad_H // self.window_size[0],
                self.window_size[0],
                self.pad_W // self.window_size[1],
                self.window_size[1],
            )
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
                self.num_windows, self.window_size[0] * self.window_size[1]
            )
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = attn.view(x.size(0) // self.num_windows, self.num_windows, self.num_heads, x.size(1), x.size(1))
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)

            return attn

    inputs = [torch.randn(64, 49, 96), torch.randn(64, 3, 49, 49)]

    model = Attn_mask_model(inplace)
    model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=inputs)

    verify(inputs, model, compiled_model)
