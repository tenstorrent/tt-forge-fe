# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from test.mlir.llama.utils.utils import load_model
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import pybuda


@pytest.mark.xfail(reason="Tile broadcast op is not supported on MLIR.")
def test_llama_inference():
    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    framework_model = load_model(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Sanity run
    generation_output = framework_model.generate(input_ids=input_ids, max_new_tokens=32)
    print(tokenizer.decode(generation_output[0]))

    # Compile the model
    compiled_model = pybuda.compile(framework_model, input_ids)