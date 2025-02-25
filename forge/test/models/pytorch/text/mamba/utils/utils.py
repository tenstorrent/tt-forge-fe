# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/state-spaces/mamba-2.8b-hf

from transformers import AutoTokenizer, MambaForCausalLM


def load_model(variant):
    model = MambaForCausalLM.from_pretrained(variant)
    model.eval()
    return model


def load_input(variant):
    prompt = "Hey how are you doing?"
    tokenizer = AutoTokenizer.from_pretrained(variant)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return [input_ids]
