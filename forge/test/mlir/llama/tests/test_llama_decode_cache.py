# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import forge
from forge.verify.verify import verify
from test.mlir.llama.utils.utils import load_model
from typing import List, Tuple
from loguru import logger
from transformers import StaticCache




@pytest.mark.parametrize(
    "model_path",
    [
        "openlm-research/open_llama_3b",  # just a string, not a tuple
        "meta-llama/Llama-3.2-1B",
        pytest.param(
            "openlm-research/open_llama_3b",
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B",
            marks=pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath"),
        ),
    ],
)
def test_llama_generation_static_cache(model_path):


    model, tokenizer = load_model(model_path, return_dict=True, use_cache=True,)

    # Input prompt
    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    batch_size, prefill_len = input_ids.shape

    # Setup static cache
    max_generated_tokens = 46
    max_cache_len = prefill_len + max_generated_tokens

    past_key_values = StaticCache(
        config=model.config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        dtype=model.dtype,
    )

    # ---------------------------
    # Prefill: run initial prompt
    # ---------------------------


    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    last_logits = outputs.logits[:, -1, :]  # logits for next token

    # ---------------------------
    # Decode loop
    # ---------------------------

    generated = input_ids.clone()
    for step in range(max_generated_tokens):
        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        # Append next token to sequence
        generated = torch.cat([generated, next_token], dim=-1)
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=torch.tensor([prefill_len + step]),
        )
        last_logits = outputs.logits[:, -1, :]

        # Optional: stop if EOS is generated
        if next_token.item() == tokenizer.eos_token_id:
            break

    # ---------------------------
    # Decode result
    # ---------------------------

    decoded = tokenizer.batch_decode(generated[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"\nPrompt: {prompt}\nCompletion: {decoded}")