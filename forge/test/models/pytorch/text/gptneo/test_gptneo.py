# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
import torch
from transformers import (
    AutoTokenizer,
    GPTNeoForCausalLM,
    GPTNeoConfig,
    GPTNeoForSequenceClassification,
)
from test.models.utils import build_module_name, Framework, Task


variants = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gptneo_causal_lm(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="gptneo", variant=variant, task=Task.CAUSAL_LM)

    record_forge_property("module_name", module_name)

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

    inputs = [inputs["input_ids"], inputs["attention_mask"]]
    compiled_model = forge.compile(Wrapper(model), sample_inputs=inputs, module_name=module_name)


variants = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gptneo_sequence_classification(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="gptneo", variant=variant, task=Task.SEQUENCE_CLASSIFICATION
    )

    record_forge_property("module_name", module_name)

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

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    compiled_model = forge.compile(Wrapper(model), sample_inputs=inputs, module_name=module_name)
