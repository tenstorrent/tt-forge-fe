# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AutoTokenizer,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    GPTNeoForSequenceClassification,
)

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

variants = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gptneo_causal_lm(record_forge_property, variant):
    if variant != "EleutherAI/gpt-neo-125M":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="gptneo", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gptneo_sequence_classification(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="gptneo",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
