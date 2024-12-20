# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
from transformers import AutoModelForCausalLM, AutoTokenizer

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
@pytest.mark.xfail(
    reason="RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph - repeat interleave"
)
@pytest.mark.nightly
def test_qwen_response(variant):
    """
    Test function for generating responses and verifying model compilation.
    """
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(variant, device_map="cpu")
    model.config.return_dict = False
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
    compiled_model = forge.compile(
        model, sample_inputs=inputs, module_name=f"pt_{variant.replace('/', '_').replace('.', '_').replace('-', '_')}"
    )
