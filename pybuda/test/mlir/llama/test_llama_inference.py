# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import pybuda


def test_llama_inference():
    # Compiler configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False

    # Load Llama 3B model and tokenizer
    model_path = "openlm-research/open_llama_3b"
    config = LlamaConfig()
    config.hidden_size = 3200
    config.intermediate_size = 8640
    config.num_hidden_layers = 26
    config.pad_token_id = 0
    config.return_dict = False
    framework_model = LlamaForCausalLM.from_pretrained(
        model_path, device_map="auto", config=config
    )
    framework_model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Sanity run
    generation_output = framework_model.generate(input_ids=input_ids, max_new_tokens=32)
    print(tokenizer.decode(generation_output[0]))

    # Compile the model
    compiled_model = pybuda.compile(framework_model, input_ids)