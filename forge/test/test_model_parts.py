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
