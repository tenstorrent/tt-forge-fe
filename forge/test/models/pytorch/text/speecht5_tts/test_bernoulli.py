# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

import forge
from forge.verify.verify import DepricatedVerifyConfig, verify


class BernoulliNet(nn.Module):
    def __init__(self):
        super(BernoulliNet, self).__init__()
        self.linear = nn.Linear(10, 10)
        self.seed = 42

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))  # Output in [0, 1] range
        torch.manual_seed(42)  # Set the same seed for PyTorch RNG
        x = torch.bernoulli(x)
        return x


# Use a fixed seed for reproducibility

model = BernoulliNet()

# seed = 42
# torch.manual_seed(seed)
input_tensor = torch.randn(1, 10)

compiled_model = forge.compile(
    model,
    sample_inputs=[input_tensor],
    verify_cfg=DepricatedVerifyConfig(verify_forge_codegen_vs_framework=True),
    module_name="bernouli",
)

# Now, propagate the seed when verifying
verify([input_tensor], model, compiled_model)
