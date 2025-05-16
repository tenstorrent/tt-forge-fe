# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AutoTokenizer,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    GPTNeoForSequenceClassification,
    GPTNeoModel,
)

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.models_utils import (
    _prepare_4d_causal_attention_mask_with_cache_position,
)
from test.utils import download_model

GPTNeoModel._prepare_4d_causal_attention_mask_with_cache_position = (
    _prepare_4d_causal_attention_mask_with_cache_position
)

variants = [
    pytest.param(
        "EleutherAI/gpt-neo-125M",
        marks=pytest.mark.xfail,
    ),
    pytest.param(
        "EleutherAI/gpt-neo-1.3B",
        marks=pytest.mark.xfail,
    ),
    pytest.param(
        "EleutherAI/gpt-neo-2.7B",
        marks=pytest.mark.skip(
            reason="Insufficient host DRAM to run this model (requires a bit more than 28 GB during compile time)"
        ),
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_gptneo_causal_lm(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="gptneo", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Set random seed for repeatability
    torch.manual_seed(42)

    # Load tokenizer and model
    # Variants: # EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B,
    # EleutherAI/gpt-neo-2.7B

    config = download_model(GPTNeoConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GPTNeoConfig(**config_dict)

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(GPTNeoForCausalLM.from_pretrained, variant, config=config)

    # Sample input text
    prompt = "My name is Bert, and I am"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, pad_to_max_length=True, truncation=True)

    # Wrapper to get around attention mask
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    framework_model = Wrapper(model)

    inputs = [inputs["input_ids"], inputs["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


variants = [
    pytest.param(
        "EleutherAI/gpt-neo-125M",
        marks=pytest.mark.skip(
            reason="Insufficient host DRAM to run this model (requires a bit more than 24 GB during compile time)"
        ),
    ),
    pytest.param(
        "EleutherAI/gpt-neo-1.3B",
        marks=pytest.mark.xfail,
    ),
    pytest.param(
        "EleutherAI/gpt-neo-2.7B",
        marks=pytest.mark.skip(
            reason="Insufficient host DRAM to run this model (requires a bit more than 28 GB during compile time)"
        ),
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_gptneo_sequence_classification(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="gptneo",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    # Variants: # EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B,
    # EleutherAI/gpt-neo-2.7B

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(GPTNeoForSequenceClassification.from_pretrained, variant, torchscript=True)

    # Load data sample
    review = "the movie was great!"

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    framework_model = Wrapper(model)

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
