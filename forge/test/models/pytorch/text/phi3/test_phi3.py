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
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import TextModelWrapper

variants = ["microsoft/phi-3-mini-4k-instruct", "microsoft/phi-3-mini-128k-instruct"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail
def test_phi3_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = Phi3ForCausalLM.from_pretrained(variant, trust_remote_code=True, use_cache=False)
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_token_classification(variant):
    if variant == "microsoft/phi-3-mini-4k-instruct":
        pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 29 GB)")
    elif variant == "microsoft/phi-3-mini-128k-instruct":
        pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 31 GB)")

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, trust_remote_code=True)
    model = Phi3ForTokenClassification.from_pretrained(variant, trust_remote_code=True, use_cache=False)
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB")
@pytest.mark.parametrize("variant", variants)
def test_phi3_sequence_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Phi3Config from pretrained variant, disable return_dict and caching.
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["use_cache"] = False
    config_dict["pad_token_id"] = None
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant, return_tensors="pt", trust_remote_code=True)
    model = Phi3ForSequenceClassification.from_pretrained(variant, trust_remote_code=True, config=config)
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # input_prompt
    input_prompt = "the movie was great!"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
