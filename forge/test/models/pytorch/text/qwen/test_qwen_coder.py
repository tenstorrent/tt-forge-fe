# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

# Variants for testing
variants = [
    pytest.param(
        "Qwen/Qwen2.5-Coder-0.5B",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "Qwen/Qwen2.5-Coder-1.5B",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 23 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "Qwen/Qwen2.5-Coder-3B",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 25 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "Qwen/Qwen2.5-Coder-7B",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        marks=[pytest.mark.xfail],
    ),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_qwen_clm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWENCODER,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    if variant == "Qwen/Qwen2.5-Coder-32B-Instruct":
        raise RuntimeError("Requires multi-chip support")

    # Load model and tokenizer
    framework_model = AutoModelForCausalLM.from_pretrained(variant, device_map="cpu")
    framework_model.config.return_dict = False
    tokenizer = AutoTokenizer.from_pretrained(variant)

    # Prepare input
    prompt = "write a quick sort algorithm."
    messages = [
        {"role": "system", "content": "You are Qwen, created by TT Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize and prepare inputs
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    inputs = [input_ids, attention_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
