# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
import forge


def test_arrange():
    class arrange_op(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_size = 1
            self.sequence_lengths = torch.tensor([5], dtype=torch.int64)

        def forward(self, logits):

            pooled_logits = logits[torch.arange(self.batch_size, device=logits.device), self.sequence_lengths]

            return pooled_logits

    inputs = [torch.randn(1, 32, 2)]

    model = arrange_op()
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
        module_name="pt_d1",
    )
