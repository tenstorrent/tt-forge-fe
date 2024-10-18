# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

import forge
from test.mlir.llama.utils.utils import load_llama32


@pytest.mark.skip(reason="No need to run in CI for now")
def test_llama_inference():
    model, tokenizer = load_llama32()

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Sanity run
    generation_output = model.generate(input_ids=input_ids, max_length=20)
    print(tokenizer.decode(generation_output[0]))

    # Compile the model
    copiled_model = forge.compile(model, input_ids)
