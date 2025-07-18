# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    AutoTokenizer,
    GPTNeoForCausalLM,
    GPTNeoForSequenceClassification,
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

from test.models.models_utils import TextModelWrapper
from test.utils import download_model

variants = [
    pytest.param(
        "EleutherAI/gpt-neo-125M",
    ),
    pytest.param(
        "EleutherAI/gpt-neo-1.3B",
    ),
    pytest.param(
        "EleutherAI/gpt-neo-2.7B",
        marks=[
            pytest.mark.out_of_memory,
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_gptneo_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GPTNEO,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )
    if variant == "EleutherAI/gpt-neo-2.7B":
        pytest.xfail(reason="Requires multi-chip support")

    # Load tokenizer and model
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(GPTNeoForCausalLM.from_pretrained, variant, use_cache=False, return_dict=True)
    framework_model = TextModelWrapper(model=model, text_embedding=model.transformer.wte)
    framework_model.eval()

    # Sample input text
    prompt = "My name is Bert, and I am"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, padding="max_length", truncation=True)

    inputs = [inputs["input_ids"], inputs["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    pytest.param(
        "EleutherAI/gpt-neo-125M",
    ),
    pytest.param(
        "EleutherAI/gpt-neo-1.3B",
    ),
    pytest.param(
        "EleutherAI/gpt-neo-2.7B",
        marks=[
            pytest.mark.out_of_memory,
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_gptneo_sequence_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GPTNEO,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    if variant == "EleutherAI/gpt-neo-2.7B":
        pytest.xfail(reason="Requires multi-chip support")

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(GPTNeoForSequenceClassification.from_pretrained, variant, use_cache=False, return_dict=True)
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # Load data sample
    prompt = "the movie was great!"

    # Data preprocessing
    input_tokens = tokenizer(prompt, return_tensors="pt")

    inputs = [input_tokens["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
