# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention
from peft import LoraConfig, get_peft_model
from transformers.utils import cached_file



def load_model(model_path="openlm-research/open_llama_3b", **kwargs):
    # Default config values
    config = LlamaConfig.from_pretrained(model_path)

    # Use defaults or values from kwargs
    config.return_dict = kwargs.get("return_dict", False)
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)
    if "num_hidden_layers" in kwargs and kwargs["num_hidden_layers"] is not None:
        config.num_hidden_layers = kwargs["num_hidden_layers"]

    # Load the model
    framework_model = LlamaForCausalLM.from_pretrained(model_path, config=config)
    framework_model.eval()

    use_lora = kwargs.get("use_lora", False)
    if use_lora:
        lora_r = kwargs.get("lora_r", 4)
        lora_alpha = kwargs.get("lora_alpha", 8)
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, task_type="CAUSAL_LM")
        framework_model = get_peft_model(framework_model, lora_config)

    # Using AutoTokenizer for default tokenizers for both openllama and llama 3.2
    use_fast = kwargs.get("use_fast", True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)

    return framework_model, tokenizer


from safetensors.torch import load_file  # or torch.load if using .bin

def load_attention_layer_only(model_path="openlm-research/open_llama_3b", layer_idx=0, dtype=torch.float32):
    # Load full config
    config = LlamaConfig.from_pretrained(model_path)
    
    # Construct a single attention layer
    attention_layer = LlamaAttention(config, layer_idx=layer_idx).to(dtype=dtype)
    attention_layer.eval()
    
    # # Download and locate the weight file from HF hub
    # bin_path = cached_file(model_path, "pytorch_model.bin")
    # weights = torch.load(bin_path, map_location="cpu")
    # prefix = f"model.layers.{layer_idx}.self_attn."
    # attn_state_dict = {
    #     k.replace(prefix, ""): v.to(dtype)
    #     for k, v in weights.items() if k.startswith(prefix)
    # }
    # attention_layer.load_state_dict(attn_state_dict, strict=False)

    return attention_layer, config
