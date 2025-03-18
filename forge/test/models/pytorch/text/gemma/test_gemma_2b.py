# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GemmaConfig,
    GemmaForCausalLM,
)

import forge
from forge.verify.verify import verify

from test.models.pytorch.text.gemma.utils.model_utils import (
    generate_no_cache,
    pad_inputs,
)
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

variants = [
    "google/gemma-2b",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b(forge_property_recorder, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 48 GB)")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="gemma",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.TEXT_GENERATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Random see for reproducibility
    torch.manual_seed(42)

    config = download_model(GemmaConfig.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = GemmaConfig(**config_dict)
    framework_model = download_model(GemmaForCausalLM.from_pretrained, variant, config=config)

    # Load tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is your favorite city?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Sanity run
    generate_ids = framework_model.generate(inputs.input_ids, max_length=30)
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
        0
    ]

    print(f"Sanity run generated text: {generated_text}")

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "google/gemma-2-2b-it",
            marks=pytest.mark.xfail(reason="RuntimeError: Trying to access element outside of dimensions: 2"),
        ),
        pytest.param(
            "google/gemma-2-9b-it",
            marks=pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 50 GB during compile time)"
            ),
        ),
    ],
)
def test_gemma_pytorch_v2(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="gemma", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

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
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Runtime and Post-Processing
    generated_text = generate_no_cache(
        max_new_tokens=max_new_tokens, model=compiled_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=tokenizer
    )
    print(generated_text)
