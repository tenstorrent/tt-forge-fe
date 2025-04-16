# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    AutoTokenizer,
    PhiForCausalLM,
    PhiForSequenceClassification,
    PhiForTokenClassification,
)

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.utils import download_model

variants = ["microsoft/phi-1_5"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail
def test_phi_1_5_causal_lm_pytorch(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="phi1.5",
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    forge_property_recorder.record_group("red")

    # Load tokenizer and model from HuggingFace
    framework_model = download_model(PhiForCausalLM.from_pretrained, variant, return_dict=False, use_cache=False)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"
    inputs = tokenizer(input_prompt, return_tensors="pt").to("cpu")

    sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs, module_name)

    # Model Verification
    verify(sample_inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail
def test_phi_1_5_token_classification_pytorch(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="phi1.5",
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    forge_property_recorder.record_group("generality")

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(
        PhiForTokenClassification.from_pretrained, variant, return_dict=False, use_cache=False
    )
    framework_model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"
    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name, forge_property_handler=forge_property_recorder)

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail
def test_phi_1_5_sequence_classification_pytorch(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="phi1.5",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    forge_property_recorder.record_group("generality")

    # Load tokenizer and model from HuggingFace
    framework_model = download_model(
        PhiForSequenceClassification.from_pretrained, variant, return_dict=False, use_cache=False
    )
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    framework_model.eval()

    # input_prompt
    input_prompt = "the movie was great!"
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )

    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name, forge_property_handler=forge_property_recorder)

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
