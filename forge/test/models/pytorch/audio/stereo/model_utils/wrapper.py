# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids}
        output = self.model(**inputs)
        return output.logits
