# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaForCausalLM

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
from test.models.pytorch.text.gemma.model_utils.model_utils import (
    generate_no_cache,
    pad_inputs,
)
from test.utils import download_model

variants = [
    "google/gemma-2b",
]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b(variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 24 GB)")

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GEMMA,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.TEXT_GENERATION,
    )

    model = download_model(GemmaForCausalLM.from_pretrained, variant, use_cache=False)
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # Load tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is your favorite city?"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
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


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "google/gemma-2-2b-it",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            "google/gemma-2-9b-it",
            marks=[pytest.mark.xfail, pytest.mark.out_of_memory],
        ),
    ],
)
def test_gemma_pytorch_v2(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GEMMA,
        variant=variant,
        task=Task.QA,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )
    if variant == "google/gemma-2-9b-it":
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant)
    framework_model = AutoModelForCausalLM.from_pretrained(variant, return_dict=False, use_cache=False)
    framework_model.eval()
    prompt = "What is the tallest mountain?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    max_new_tokens = 200
    padded_inputs, seq_len = pad_inputs(input_ids, max_new_tokens)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model)

    # Runtime and Post-Processing
    generated_text = generate_no_cache(
        max_new_tokens=max_new_tokens, model=compiled_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=tokenizer
    )
    print(generated_text)


variants = [
    "google/gemma-2-27b-it",
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_gemma_pytorch_27b(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.FLUX,
        variant=variant,
        task=Task.QA,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")
