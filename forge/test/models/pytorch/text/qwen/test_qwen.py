# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "Qwen/Qwen1.5-0.5B",
            marks=[pytest.mark.xfail],
        ),
    ],
)
def test_qwen1_5_causal_lm(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="qwen1.5", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Setup model configuration
    config = Qwen2Config.from_pretrained(variant)
    config.use_cache = False
    config.return_dict = False

    # Load model and tokenizer with config
    framework_model = Qwen2ForCausalLM.from_pretrained(variant, config=config)
    tokenizer = Qwen2Tokenizer.from_pretrained(variant)
    tokenizer.pad_token, tokenizer.pad_token_id = (tokenizer.eos_token, tokenizer.eos_token_id)

    # Disable DynamicCache
    # See: https://github.com/tenstorrent/tt-buda/issues/42
    framework_model._supports_cache_class = False

    # Example usage
    batch_size = 1
    prompt = ["My name is Jim Keller and"] * batch_size

    inputs = tokenizer(prompt)

    input_ids = torch.tensor(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"])

    inputs = [input_ids, attention_mask]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["Qwen/Qwen1.5-0.5B-Chat"])
def test_qwen1_5_chat(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="qwen1.5", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Setup model configuration
    config = Qwen2Config.from_pretrained(variant)
    config.use_cache = False
    config.return_dict = False

    # Load model and tokenizer with config
    framework_model = Qwen2ForCausalLM.from_pretrained(variant, config=config)
    tokenizer = Qwen2Tokenizer.from_pretrained(variant)
    tokenizer.pad_token, tokenizer.pad_token_id = (tokenizer.eos_token, tokenizer.eos_token_id)

    # Disable DynamicCache
    # See: https://github.com/tenstorrent/tt-buda/issues/42
    framework_model._supports_cache_class = False

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

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
