# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

import forge
from forge.verify.verify import verify, verify_backward


class MNISTLinear(nn.Module):
    def __init__(
        self, input_size=784, output_size=10, hidden_size=512, bias=True, dtype=torch.float32
    ):  # changed hidden_size to 512 because matmul 256 x batch_size is not supported in ttnn
        super(MNISTLinear, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=bias, dtype=dtype),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def test_mnist(train=False):
    batch_size = 128
    fw_model = MNISTLinear()
    fw_model.eval() if not train else fw_model.train()

    inputs = torch.randn(batch_size, 784)

    co_model = forge.compile(fw_model, sample_inputs=inputs)

    fw_out, co_out = verify([inputs], fw_model, co_model)
    if train:
        grads = torch.randn_like(fw_out)
        verify_backward([inputs], grads, fw_out, co_out, fw_model, co_model)
