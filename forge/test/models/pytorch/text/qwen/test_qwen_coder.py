# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
from transformers import AutoModelForCausalLM, AutoTokenizer
from test.models.utils import build_module_name, Framework, Task
from forge.verify.verify import verify


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
@pytest.mark.model_analysis
@pytest.mark.nightly
def test_qwen_clm(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="qwen_coder", variant=variant, task=Task.CAUSAL_LM
    )

    record_forge_property("module_name", module_name)

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

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)
