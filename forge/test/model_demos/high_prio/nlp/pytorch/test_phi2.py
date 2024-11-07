# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import (
    PhiForCausalLM,
    AutoTokenizer,
    PhiConfig,
    PhiForTokenClassification,
    PhiForSequenceClassification,
)
import pytest
import forge

variants = ["microsoft/phi-2", "microsoft/phi-2-pytdml"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_phi2_clm(variant, test_device):

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = PhiConfig(**config_dict)

    # Load model and tokenizer from HuggingFace
    model = PhiForCausalLM.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # input_prompt
    input_prompt = "Write a detailed analogy between mathematics and a lighthouse."

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"].to(torch.int32)
    attn_mask = inputs["attention_mask"].to(torch.float32)

    inputs = [input_ids, attn_mask]
    compiled_model = forge.compile(
        model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_")) + "_causal_lm"
    )


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi2_token_classification(variant, test_device):

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = PhiConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    model = PhiForTokenClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    compiled_model = forge.compile(
        model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_")) + "_token_cls"
    )


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi2_sequence_classification(variant, test_device):

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config_dict["pad_token_id"] = None
    config = PhiConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    model = PhiForSequenceClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    model.eval()

    # input_prompt
    input_prompt = "I am not satisfied with the quality of this product."

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]
    compiled_model = forge.compile(
        model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_")) + "_seq_cls"
    )
