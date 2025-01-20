# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

import forge


def test_conv2d(test_device):
    class Conv2dModule(nn.Module):
        def __init__(self):
            super(Conv2dModule, self).__init__()
            self.conv = nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=0,
                dilation=(1, 1),
                groups=1,
            )

        def forward(self, x):
            x = torch.nn.functional.pad(x, (0, 1, 0, 1))
            out = self.conv(x)

            return out

    model = Conv2dModule()

    input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    output_tensor = model(input_tensor)
    print(output_tensor.shape)
    compiled_model = forge.compile(model, sample_inputs=[input_tensor], module_name="conv2d")
