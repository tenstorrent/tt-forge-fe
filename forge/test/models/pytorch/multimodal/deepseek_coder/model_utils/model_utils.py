# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


class DeepSeekWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_tokens = model.model.embed_tokens

    def forward(self, input_tensor, attention_mask=None, past_key_values=None):
        inputs_embeds = self.embed_tokens(input_tensor)
        past_key_values_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_tensor.shape, inputs_embeds, past_key_values_length
        )
        return self.model(input_ids=input_tensor, attention_mask=causal_attention_mask).logits
