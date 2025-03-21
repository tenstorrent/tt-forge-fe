# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 26 GB)")
def test_ministral(record_forge_property):
    # hf_token = ""

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="ministral-3b-instruct",
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority_1")
    forge_property_recorder.record_model_name(module_name)

    # Load and modify the model configuration
    config = AutoConfig.from_pretrained("ministral/Ministral-3b-instruct", use_auth_token=hf_token)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ministral/Ministral-3b-instruct", use_auth_token=hf_token)

    # Load model with modified configuration
    framework_model = AutoModelForCausalLM.from_pretrained(
        "ministral/Ministral-3b-instruct", config=config, use_auth_token=hf_token
    )
    framework_model.eval()

    # Generate sample inputs
    prompt = "What are the benefits of AI in healthcare?"
    input_tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = input_tokens["input_ids"]
    inputs = [input_ids]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
