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


import pytest
import torch
import torch.nn as nn
import forge
from forge.verify.verify import verify
import onnx


def test_scatter_nd():
    class scatter_nd(nn.Module):
        def __init__(self):
            super().__init__()
            self.mask_length = 256
            self.min_dtype = torch.finfo(torch.float32).min

        def forward(self, causal_mask, padding_mask):
            causal_mask[:, :, :, : self.mask_length] = causal_mask[:, :, :, : self.mask_length].masked_fill(
                padding_mask, self.min_dtype
            )
            return causal_mask

    model = scatter_nd()
    model.eval()

    # Create inputs
    torch.manual_seed(0)
    causal_mask = torch.tril(torch.ones((1, 1, 256, 256), dtype=torch.float32)) * 0
    padding_mask = torch.zeros((1, 1, 256, 256), dtype=torch.bool)

    padding_mask[0, 0, -3:, -3:] = True

    inputs = [causal_mask.clone(), padding_mask.clone()]

    # Export model to ONNX
    onnx_path = "phi2_s.onnx"
    torch.onnx.export(model, (inputs[0], inputs[1]), onnx_path, opset_version=17, verbose=True)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule("phi2_sanity", onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
