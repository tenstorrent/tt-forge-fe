# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch


class GPT_OSS_Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_tokens = model.model.embed_tokens

    def forward(self, input_tensor):
        outputs = self.model(input_ids=input_tensor)
        return outputs.logits
