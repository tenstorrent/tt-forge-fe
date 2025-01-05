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
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name

variants = ["microsoft/phi-2", "microsoft/phi-2-pytdml"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_phi2_clm(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="phi2", variant=variant, task="clm")

    record_forge_property("module_name", module_name)

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

    # Sanity
    fw_out = model(*inputs)

    # Inference
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_phi2_token_classification(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="phi2", variant=variant, task="token_cls")

    record_forge_property("module_name", module_name)

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

    # Sanity
    fw_out = model(*inputs)

    # Inference
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_phi2_sequence_classification(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="phi2", variant=variant, task="seqcls")

    record_forge_property("module_name", module_name)

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

    # Sanity
    fw_out = model(*inputs)

    # Inference
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden(golden=fw_out, calculated=co_out[0], pcc=0.99)
