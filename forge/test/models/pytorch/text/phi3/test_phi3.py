# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    AutoTokenizer,
    Phi3Config,
    Phi3ForCausalLM,
    Phi3ForSequenceClassification,
    Phi3ForTokenClassification,
)

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["microsoft/phi-3-mini-4k-instruct"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_causal_lm(forge_property_recorder, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 64 GB)")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="phi3", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    if variant in ["microsoft/phi-3-mini-4k-instruct"]:
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    framework_model = Phi3ForCausalLM.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"

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
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_token_classification(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="phi3",
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)

    framework_model = Phi3ForTokenClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name, forge_property_handler=forge_property_recorder)

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_sequence_classification(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="phi3",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config_dict["pad_token_id"] = None
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    framework_model = Phi3ForSequenceClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model.eval()

    # input_prompt
    input_prompt = "the movie was great!"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name, forge_property_handler=forge_property_recorder)

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
