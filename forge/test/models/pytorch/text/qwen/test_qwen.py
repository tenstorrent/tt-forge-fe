# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import TextModelWrapper


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
def test_qwen1_5_causal_lm(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWEN15,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer with config
    model = Qwen2ForCausalLM.from_pretrained(variant, use_cache=False)
    # Disable DynamicCache
    # See: https://github.com/tenstorrent/tt-buda/issues/42
    model._supports_cache_class = False
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()
    tokenizer = Qwen2Tokenizer.from_pretrained(variant)
    tokenizer.pad_token, tokenizer.pad_token_id = (tokenizer.eos_token, tokenizer.eos_token_id)

    # Example usage
    batch_size = 1
    prompt = ["My name is Jim Keller and"] * batch_size

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Get input_ids and attention_mask
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    inputs = [input_ids, attention_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["Qwen/Qwen1.5-0.5B-Chat"])
def test_qwen1_5_chat(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWEN15,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer with config
    model = Qwen2ForCausalLM.from_pretrained(variant, use_cache=False)
    # Disable DynamicCache
    # See: https://github.com/tenstorrent/tt-buda/issues/42
    model._supports_cache_class = False
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()
    tokenizer = Qwen2Tokenizer.from_pretrained(variant)
    tokenizer.pad_token, tokenizer.pad_token_id = (tokenizer.eos_token, tokenizer.eos_token_id)

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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
