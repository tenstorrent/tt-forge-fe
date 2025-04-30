# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

import forge
from forge.verify.verify import verify


def test_resize():
    class resize(nn.Module):
        def __init__(self):
            super().__init__()
            self.new_patch_heigth = 32
            self.new_patch_width = 42

        def forward(self, patch_pos_embed):
            y = nn.functional.interpolate(
                patch_pos_embed, size=(self.new_patch_heigth, self.new_patch_width), mode="bicubic", align_corners=False
            )
            return y

    framework_model = resize()
    framework_model.eval()

    inputs = torch.randn(1, 192, 50, 83)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
