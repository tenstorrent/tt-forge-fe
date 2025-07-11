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
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

variants = ["ministral/Ministral-3b-instruct"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_ministral_3b(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MINISTRAL,
        variant=variant,
        task=Task.NLP_TEXT_GEN,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(variant)
    # Load model with modified configuration
    framework_model = AutoModelForCausalLM.from_pretrained(variant)
    framework_model.eval()

    # Generate sample inputs
    prompt = "What are the benefits of AI in healthcare?"
    input_tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = input_tokens["input_ids"]
    inputs = [input_ids]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
