# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

import forge


def load_model(model_path="openlm-research/open_llama_3b", **kwargs):
    # Default config values
    config = LlamaConfig.from_pretrained(model_path)
    config.hidden_size = 3200
    config.intermediate_size = 8640
    config.num_hidden_layers = 26
    config.pad_token_id = 0

    # Use defaults or values from kwargs
    config.return_dict = kwargs.get("return_dict", False)
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)

    # Load the model
    framework_model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", config=config)
    framework_model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    return framework_model, tokenizer


def load_llama32(model_id: str = "meta-llama/Llama-3.2-1B", **kwargs):

    # Default token
    token = kwargs.get("token", "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")

    # Load the config
    config = LlamaForCausalLM.config_class.from_pretrained(model_id, token=token)
    config.return_dict = kwargs.get("return_dict", False)
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, use_fast=True)

    # Load the model
    model = LlamaForCausalLM.from_pretrained(model_id, token=token, config=config)
    model.eval()

    return model, tokenizer
