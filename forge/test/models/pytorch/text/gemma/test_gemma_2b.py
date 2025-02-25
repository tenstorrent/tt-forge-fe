# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoTokenizer, GemmaConfig, GemmaForCausalLM

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

variants = [
    "google/gemma-2b",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_gemma_2b(record_forge_property, variant):
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
    record_forge_property("tags.model_name", module_name)

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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
