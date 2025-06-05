# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model

variants = ["mistralai/Ministral-8B-Instruct-2410"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Transient test - Out of memory due to other tests in CI pipeline")
def test_ministral_8b(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MINISTRAL,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CAUSAL_LM,
        group=ModelGroup.RED,
    )

    # Load model and tokenizer
    framework_model = download_model(AutoModelForCausalLM.from_pretrained, variant, return_dict=False, use_cache=False)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input
    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
