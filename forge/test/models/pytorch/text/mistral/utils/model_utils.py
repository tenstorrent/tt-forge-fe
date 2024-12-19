# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch


class BaseModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor, *kv):
        """
        input_ids: Shape [bs, 1]
        attention_mask: Shape [bs, seqlen]
        position_ids: Shape [1, 1]
        kv: KV cache in format (k0, v0, k1, v1, ..., k_{L-1}, v_{L-1}) where L is the number of layers/blocks
        """
        kv = tuple(zip(kv[:-1:2], kv[1::2]))  # making tuple of pairs (key_cache, value_cache)
        outputs = self.model(input_ids, attention_mask, position_ids, kv)
        # flattening past key values because TT compiler expects flattened output in format tuple(torch.Tensor,  ..., torch.Tensor)
        outputs = [outputs[0]] + [el for subl in outputs[1] for el in subl]
        return tuple(outputs)


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next.unsqueeze(0)
