# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from forge.verify.verify import verify
import forge
import torch.nn as nn

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(in_features=768, out_features=1)

    def forward(self, x):
        y = self.model(x)
        return y

def test_sanity():

    framework_model = Wrapper()

    inputs = [torch.randn(1,768)]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

   