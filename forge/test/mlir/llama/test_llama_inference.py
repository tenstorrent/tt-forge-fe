# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import forge
from test.mlir.llama.utils.utils import load_model


@pytest.mark.xfail()
def test_llama_inference():
    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model = load_model(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Sanity run
    generation_output = framework_model.generate(input_ids=input_ids, max_new_tokens=32)
    print(tokenizer.decode(generation_output[0]))

    # Compile the model
    compiled_model = forge.compile(framework_model, input_ids)


@pytest.mark.skip(reason="No need to run in CI, this is PoC that should be mapped to work on device.")
def test_llama_inference_no_cache_cpu():
    """
    This function tests the inference of the Llama 3B model without using a past-cache (KV cache).
    It generates text token by token, which can slow down over time as the model has to compute
    all key-value (KV) pairs for each new token. The function demonstrates how to load the model
    and tokenizer, prepare an input prompt, and generate a sequence of tokens until a specified
    maximum number of new tokens is reached or an end-of-sequence token is encountered.
    """
    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model = load_model(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    max_new_tokens = 46
    generated_tokens = input_ids
    for i in range(max_new_tokens):
        logits = framework_model(input_ids)
        next_token_logits = logits[0][:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(generated_text)


@pytest.mark.skip(reason="No need to run in CI, this is PoC that should be mapped to work on device.")
def test_llama_inference_cache_cpu():
    """
    This function tests the inference of the Llama 3B model using a past-cache (KV cache).
    By utilizing cached key-value (KV) pairs, the model can generate text more efficiently
    as it doesn't need to recompute KV pairs for previously generated tokens. The function
    demonstrates how to load the model and tokenizer, prepare an input prompt, and generate
    a sequence of tokens until a specified maximum number of new tokens is reached or an
    end-of-sequence token is encountered.

    Steps:
    1. Load the Llama 3B model and tokenizer with caching enabled.
    2. Prepare an input prompt and convert it to input IDs.
    3. Initialize past key-values and other necessary inputs.
    4. Perform a prefill step to get the initial logits and past key-values.
    5. Generate tokens iteratively, updating the past key-values and input IDs.
    6. Decode the generated tokens into text and print the result.
    """
    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model = load_model(model_path, use_cache=True)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Prepare input sentence
    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_length = input_ids.size(-1)

    # Prepare other inputs
    past_key_values = None
    pkv_length = len(past_key_values[0][0].shape[2]) if past_key_values else 0
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(pkv_length, seq_length + pkv_length, dtype=torch.long).unsqueeze(0)

    inputs = (input_ids, attention_mask, position_ids, past_key_values)

    # Prefill
    logits, past_key_values = framework_model(*inputs)

    # Fetch token for 0 iteration
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)

    max_new_tokens = 46
    generated_tokens = input_ids
    for i in range(max_new_tokens):
        logits, past_key_values = framework_model(input_ids=next_token.unsqueeze(0), past_key_values=past_key_values)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break
        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    # Generated text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(generated_text)
