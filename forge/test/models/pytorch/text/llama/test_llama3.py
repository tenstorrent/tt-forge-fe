# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Llama3 Demo - CasualLM

import pytest
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
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

variants = [
    pytest.param(
        "meta-llama/Meta-Llama-3-8B",
        marks=[
            pytest.mark.skip(
                "Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        marks=[
            pytest.mark.skip(
                "Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "meta-llama/Llama-3.1-8B",
        marks=[
            pytest.mark.skip(
                "Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "meta-llama/Llama-3.1-8B-Instruct",
        marks=[
            pytest.mark.xfail,
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        marks=[
            pytest.mark.xfail,
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param("meta-llama/Llama-3.2-1B"),
    pytest.param("meta-llama/Llama-3.2-1B-Instruct"),
    pytest.param(
        "meta-llama/Llama-3.2-3B",
        marks=[
            pytest.mark.skip(
                "Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "meta-llama/Llama-3.2-3B-Instruct",
        marks=[
            pytest.mark.xfail,
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "huggyllama/llama-7b",
        marks=[
            pytest.mark.skip(
                "Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param("meta-llama/Meta-Llama-3.1-70B", marks=pytest.mark.xfail),
    pytest.param("meta-llama/Meta-Llama-3.1-70B-Instruct", marks=pytest.mark.xfail),
    pytest.param("meta-llama/Llama-3.3-70B-Instruct", marks=pytest.mark.xfail),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_llama3_causal_lm(variant):
    if variant in [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ]:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.LLAMA3,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=group,
        priority=priority,
    )

    if variant in [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model (with tokenizer)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = download_model(AutoModelForCausalLM.from_pretrained, variant, use_cache=False)
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # Input prompt
    input_prompt = "Hey how are you doing today?"
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    pytest.param(
        "meta-llama/Meta-Llama-3-8B",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "meta-llama/Llama-3.1-8B",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "meta-llama/Llama-3.1-8B-Instruct",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param("meta-llama/Llama-3.2-1B"),
    pytest.param("meta-llama/Llama-3.2-1B-Instruct"),
    pytest.param(
        "meta-llama/Llama-3.2-3B",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 25 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "meta-llama/Llama-3.2-3B-Instruct",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 26 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "huggyllama/llama-7b",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_llama3_sequence_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.LLAMA3,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model (with tokenizer)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(AutoModelForSequenceClassification.from_pretrained, variant, use_cache=False)
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # Input prompt
    input_prompt = "Movie is great"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    inputs = [input_ids]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_value = co_out[0].argmax(-1).item()

    print(f"Prediction : {model.config.id2label[predicted_value]}")
