# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch


class RobertaWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor, attention_mask):
        outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask)
        return outputs.logits
