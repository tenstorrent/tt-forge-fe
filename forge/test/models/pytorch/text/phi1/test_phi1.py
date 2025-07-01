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

variants = ["microsoft/phi-1", "microsoft/phi-1_5"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi1_causal_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI1_5,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load tokenizer and model from HuggingFace
    model = download_model(PhiForCausalLM.from_pretrained, variant, use_cache=False)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
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
def test_phi1_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI1_5,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(PhiForTokenClassification.from_pretrained, variant, use_cache=False)

    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"
    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_token_class_ids = co_out[0].argmax(-1)[0]
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids]

    print(f"Context: {input_prompt}")
    print(f"Answer: {predicted_tokens_classes}")


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi1_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI1_5,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    model = download_model(PhiForSequenceClassification.from_pretrained, variant, use_cache=False)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)

    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # input_prompt
    input_prompt = "the movie was great!"
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_value = co_out[0].argmax(-1).item()
    print(f"Predicted Sentiment: {model.config.id2label[predicted_value]}")
