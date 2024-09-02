# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import forge

def load_model(model_path="openlm-research/open_llama_3b", use_cache=False):
    # Compiler configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False

    # Load Llama 3B model
    config = LlamaConfig()
    config.hidden_size = 3200
    config.intermediate_size = 8640
    config.num_hidden_layers = 26
    config.pad_token_id = 0
    config.return_dict = False
    config.use_cache = use_cache
    framework_model = LlamaForCausalLM.from_pretrained(
        model_path, device_map="auto", config=config
    )
    framework_model.eval()
    
    return framework_model
