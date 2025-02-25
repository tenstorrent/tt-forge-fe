# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model():
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
    model.config.use_cache = False
    model.eval()
    return model


def load_input():
    test_input = "This is a sample text from "
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1", padding_side="left")
    inputs = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )
    return [inputs["input_ids"], inputs["attention_mask"]]
