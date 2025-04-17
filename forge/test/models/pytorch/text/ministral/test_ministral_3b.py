# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

variants = ["ministral/Ministral-3b-instruct"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.xfail
def test_ministral_3b(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_propertiese(
        framework=Framework.PYTORCH,
        model="ministral",
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    forge_property_recorder.record_group("red")

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
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
