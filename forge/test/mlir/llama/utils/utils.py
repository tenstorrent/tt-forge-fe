# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import forge


def load_model_and_tokenizer(**kwargs):
    # Default config values
    config = LlamaConfig()
    config.hidden_size = 3200
    config.intermediate_size = 8640
    config.num_hidden_layers = 26
    config.pad_token_id = 0

    # Use defaults or values from kwargs
    model_path = kwargs.get("model_path", "openlm-research/open_llama_3b")
    config.return_dict = kwargs.get("return_dict", False)
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)

    # Load the model
    framework_model = LlamaForCausalLM.from_pretrained(
        model_path, device_map="auto", config=config
    )
    framework_model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    return framework_model, tokenizer
