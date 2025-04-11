# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class CogitoWrapper(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits


def get_input_model(variant):
    model = CogitoWrapper(variant)
    model.eval()

    # Tokenize chat prompt
    prompt = "Give me a short introduction to LLMs."
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": prompt},
    ]
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    model_inputs = model.tokenizer([text], return_tensors="pt")
    input_tensor_list = [model_inputs["input_ids"]]

    return input_tensor_list, model
