# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn
import torch.onnx
import onnx
import math
from forge.verify.verify import verify
import forge


@pytest.mark.parametrize(
    "use_param, model_name, onnx_file",
    [
        (True, "swin2_sanity_param", "swin_sanity_param.onnx"),
        (False, "swin2_sanity_tensor", "swin_sanity_tensor.onnx"),
    ],
)
def test_clamp(use_param, model_name, onnx_file):
    class Clamp(nn.Module):
        def __init__(self):
            super().__init__()
            log_value = torch.log(10 * torch.ones((3, 1, 1)))
            if use_param:
                self.logit_scale = nn.Parameter(log_value)
            else:
                self.logit_scale = torch.tensor(log_value)

        def forward(self, attn):
            logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
            return attn

    model = Clamp()
    model.eval()

    attn = torch.randn(64, 3, 64, 64)
    inputs = [attn]

    # Export model to ONNX
    torch.onnx.export(model, (inputs[0],), onnx_file, opset_version=17, verbose=True)

    # Load ONNX model
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    framework_model = forge.OnnxModule(model_name, onnx_model, onnx_file)

    # Compile and verify
    compiled_model = forge.compile(onnx_model, inputs)
    verify(inputs, framework_model, compiled_model)
