# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "n, m, dtype",
    [
        (4, 4, torch.float32),
        (6, 7, torch.int32),
        (8, 8, torch.bool),
    ],
)
@pytest.mark.push
def test_eye(forge_property_recorder, n, m, dtype):
    class EyeModel(torch.nn.Module):
        def __init__(self, n, m, dtype):
            super().__init__()
            self.n = n
            self.m = m
            self.dtype = dtype

        def forward(self, dummy):
            x = dummy + 1
            if self.m is None:
                out = torch.eye(self.n, dtype=self.dtype)
            else:
                out = torch.eye(self.n, self.m, dtype=self.dtype)
            return out + x * 0

    input = torch.randn(1)
    inputs = [input]

    model = EyeModel(n, m, dtype)
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)
    verify(inputs, model, compiled_model, forge_property_handler=forge_property_recorder)
