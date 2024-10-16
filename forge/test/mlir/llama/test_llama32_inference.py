# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import pipeline, AutoTokenizer, LlamaForCausalLM

import forge


def test_llama_inference():

    model_id = "meta-llama/Llama-3.2-1B"
    token = "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"

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, use_fast=True)

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Sanity run
    generation_output = model.generate(input_ids=input_ids, max_length=20)
    print(tokenizer.decode(generation_output[0]))

    # Compile the model
    copiled_model = forge.compile(model, input_ids)
