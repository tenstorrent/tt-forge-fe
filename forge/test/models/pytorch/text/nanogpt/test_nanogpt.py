# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

import forge
from forge.verify.verify import verify

variants = [
    "FinancialSupport/NanoGPT",
]


# Wrapper to get around attention mask
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, None, attention_mask)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_nanogpt_test_generation(variant):
    tokenizer = AutoTokenizer.from_pretrained(variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(variant, ignore_mismatched_sizes=True, use_cache=False, return_dict=False)

    # Input prompt
    input_prompt = "The financial market showed signs of volatility"

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=150,
        pad_to_max_length=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    # Get Inputs
    input_ids = input_ids.to(torch.int32)
    attn_mask = attn_mask.to(torch.float32)
    inputs = [input_ids, attn_mask]

    framework_model = Wrapper(model)

    compiled_model = forge.compile(
        framework_model,
        inputs,
        "pt_FinancialSupport_NanoGPT",
    )

    verify(inputs, framework_model, compiled_model)
