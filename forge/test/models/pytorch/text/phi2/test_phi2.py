# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    AutoTokenizer,
    PhiConfig,
    PhiForCausalLM,
    PhiForSequenceClassification,
    PhiForTokenClassification,
    PhiModel,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
)
from forge.verify.verify import verify

from test.models.models_utils import (
    _prepare_4d_causal_attention_mask_with_cache_position,
)

PhiModel._prepare_4d_causal_attention_mask_with_cache_position = _prepare_4d_causal_attention_mask_with_cache_position


variants = [
    pytest.param(
        "microsoft/phi-2",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "microsoft/phi-2-pytdml",
        marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 29 GB"),
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi2_clm(variant):
    if variant in ["microsoft/phi-2"]:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="phi2",
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=group,
        priority=priority,
    )

    # Load PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = PhiConfig(**config_dict)

    # Load model and tokenizer from HuggingFace
    framework_model = PhiForCausalLM.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model.eval()
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

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    pytest.param(
        "microsoft/phi-2",
        marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB"),
    ),
    pytest.param(
        "microsoft/phi-2-pytdml",
        marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 28 GB"),
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi2_token_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="phi2",
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = PhiConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    framework_model = PhiForTokenClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    pytest.param(
        "microsoft/phi-2",
        marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 27 GB"),
    ),
    pytest.param(
        "microsoft/phi-2-pytdml",
        marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 28 GB"),
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi2_sequence_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="phi2",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config_dict["pad_token_id"] = None
    config = PhiConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    framework_model = PhiForSequenceClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model.eval()

    # input_prompt
    input_prompt = "I am not satisfied with the quality of this product."

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
