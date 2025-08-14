# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from torch import nn

import forge
from forge.verify.verify import verify


def test_resize():
    class resize(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = F.interpolate(x, size=(7, 7), mode="nearest")
            return y

    a = torch.rand(1, 960, 3, 3)
    inputs = [a]

    framework_model = resize()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
