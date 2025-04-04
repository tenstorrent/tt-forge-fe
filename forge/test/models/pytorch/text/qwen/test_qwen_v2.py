# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2ForTokenClassification,
)

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

# Variants for testing
variants = [
    pytest.param(
        "Qwen/Qwen2.5-0.5B",
        marks=[pytest.mark.xfail],
    ),
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_qwen_clm(forge_property_recorder, variant):
    if variant != "Qwen/Qwen2.5-0.5B":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="qwen_v2", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    if variant in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]:
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load model and tokenizer
    framework_model = AutoModelForCausalLM.from_pretrained(variant, device_map="cpu")
    framework_model.config.return_dict = False
    tokenizer = AutoTokenizer.from_pretrained(variant)

    # Prepare input
    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize and generate
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    inputs = [input_ids, attention_mask]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["Qwen/Qwen2-7B"])
def test_qwen2_token_classification(forge_property_recorder, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 32 GB during compile time)")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="qwen_v2",
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load model and tokenizer
    framework_model = Qwen2ForTokenClassification.from_pretrained(variant)
    framework_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(variant)

    # Prepare input
    text = "HuggingFace is a company based in Paris and New York."
    model_inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")

    inputs = [model_inputs["input_ids"], model_inputs["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
