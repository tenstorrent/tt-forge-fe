# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
from transformers import AutoModelForCausalLM, AutoTokenizer


# Variants for testing
variants = ["Qwen/Qwen2.5-0.5B"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_qwen_response(variant):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(variant, device_map="cpu")
    model.config.return_dict = False
    model.config.use_cache = False
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
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_Qwen")
