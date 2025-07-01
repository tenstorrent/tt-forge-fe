# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        "gpt2",
    ],
)
def test_gpt2_text_gen(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GPT,
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    config = GPT2Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GPT2Config(**config_dict)
    model = download_model(GPT2LMHeadModel.from_pretrained, variant, config=config)

    input_ids = torch.cat(
        [torch.randint(1, model.config.vocab_size, (1, 255)), torch.zeros(1, 1, dtype=torch.int64)], dim=-1
    ).to(torch.int64)
    inputs = [input_ids]

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        "mnoukhov/gpt2-imdb-sentiment-classifier",
    ],
)
def test_gpt2_sequence_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GPT,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, padding_side="left")
    model = download_model(
        AutoModelForSequenceClassification.from_pretrained, variant, return_dict=False, use_cache=False
    )
    model.eval()

    # Prepare input
    test_input = "This is a sample text from "
    input_tokens = tokenizer(test_input, return_tensors="pt")
    inputs = [input_tokens["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, model, compiled_model)

    # post processing
    predicted_value = co_out[0].argmax(-1).item()
    print(f"Predicted Sentiment: {model.config.id2label[predicted_value]}")
