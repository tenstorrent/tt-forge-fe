# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
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
from test.utils import download_model

variants = ["microsoft/phi-4"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi_4_causal_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI4,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load tokenizer and model from HuggingFace
    model = download_model(AutoModelForCausalLM.from_pretrained, variant, use_cache=False)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"

    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs, module_name)

    # Model Verification
    verify(sample_inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail
def test_phi_4_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI4,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Test is killed at consteval compilation stage")

    # Load tokenizer and model from HuggingFace
    model = download_model(AutoModelForTokenClassification.from_pretrained, variant, use_cache=False)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"
    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_phi_4_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI4,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Test is killed at consteval compilation stage")

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(AutoModelForSequenceClassification.from_pretrained, variant, use_cache=False)
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # input_prompt
    input_prompt = "the movie was great!"
    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
