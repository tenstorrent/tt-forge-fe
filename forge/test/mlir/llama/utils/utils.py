# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

import forge


def load_model(model_path="openlm-research/open_llama_3b", **kwargs):
    # Default config values
    config = LlamaConfig.from_pretrained(model_path)

    # Use defaults or values from kwargs
    config.return_dict = kwargs.get("return_dict", False)
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)
    if "num_hidden_layers" in kwargs:
        config.num_hidden_layers = kwargs["num_hidden_layers"]

    # Load the model
    framework_model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", config=config)
    framework_model.eval()

    # Using AutoTokenizer for default tokenizers for both openllama and llama 3.2
    use_fast = kwargs.get("use_fast", True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)

    return framework_model, tokenizer
