# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

# Variants for testing
variants = [
    "Qwen/Qwen2.5-Coder-0.5B",
    "Qwen/Qwen2.5-Coder-1.5B",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_qwen_clm(record_forge_property, variant):
    if variant != "Qwen/Qwen2.5-Coder-0.5B":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="qwen_coder", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

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
