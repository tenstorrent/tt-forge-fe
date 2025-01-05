# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import forge
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer
import torch
import re
from test.models.utils import build_module_name, Framework, Task


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_qwen1_5_causal_lm(record_forge_property):
    variant = "Qwen/Qwen1.5-0.5B"

    module_name = build_module_name(framework=Framework.PYTORCH, model="qwen1.5", variant=variant, task=Task.CAUSAL_LM)

    record_forge_property("module_name", module_name)

    # Setup model configuration
    config = Qwen2Config.from_pretrained(variant)
    config.use_cache = False
    config.return_dict = False

    # Load model and tokenizer with config
    model = Qwen2ForCausalLM.from_pretrained(variant, config=config)
    tokenizer = Qwen2Tokenizer.from_pretrained(variant)
    tokenizer.pad_token, tokenizer.pad_token_id = (tokenizer.eos_token, tokenizer.eos_token_id)

    # Disable DynamicCache
    # See: https://github.com/tenstorrent/tt-buda/issues/42
    model._supports_cache_class = False

    # Example usage
    batch_size = 1
    prompt = ["My name is Jim Keller and"] * batch_size

    inputs = tokenizer(prompt)

    input_ids = torch.tensor(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"])

    inputs = [input_ids, attention_mask]

    # Pass the tensors to the model
    op = model(input_ids, attention_mask)

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


def parse_chat_completion(text: str):
    pattern = r"<\|im_start\|>\s*(\w+)\s*([\s\S]*?)\s*(?:<\|im_end\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    messages = []
    for role, content in matches:
        messages.append({"role": role, "content": content.strip()})

    return messages


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_qwen1_5_chat(record_forge_property):
    variant = "Qwen/Qwen1.5-0.5B-Chat"

    module_name = build_module_name(framework=Framework.PYTORCH, model="qwen1.5", variant=variant)

    record_forge_property("module_name", module_name)

    # Setup model configuration
    config = Qwen2Config.from_pretrained(variant)
    config.use_cache = False
    config.return_dict = False

    # Load model and tokenizer with config
    model = Qwen2ForCausalLM.from_pretrained(variant, config=config)
    tokenizer = Qwen2Tokenizer.from_pretrained(variant)
    tokenizer.pad_token, tokenizer.pad_token_id = (tokenizer.eos_token, tokenizer.eos_token_id)

    # Disable DynamicCache
    # See: https://github.com/tenstorrent/tt-buda/issues/42
    model._supports_cache_class = False

    batch_size = 1
    # Sample chat messages
    batch_messages = [
        [
            {"role": "system", "content": "You are Jim Keller, the CEO of Tenstorrent"},
            {"role": "user", "content": "Introduce yourself please!"},
        ]
        * batch_size
    ]

    # Apply chat template to each batch
    chat_texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in batch_messages[:batch_size]
    ]

    # tokenize the generated chat texts
    tokenized_inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get input_ids and attention_mask
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]

    inputs = [input_ids, attention_mask]

    # Pass the tensors to the model
    op = model(input_ids, attention_mask)

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
