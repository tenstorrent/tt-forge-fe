# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FalconForCausalLM

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["tiiuae/falcon-7b-instruct"])
def test_falcon(record_forge_property, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 32 GB)")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="falcon", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    tokenizer = AutoTokenizer.from_pretrained(variant)
    model = FalconForCausalLM.from_pretrained(variant)
    model.config.use_cache = False
    model.config.return_dict = False

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    framework_model = Wrapper(model)
    input_tokens = tokenizer("Hello, my dog is cute", return_tensors="pt")

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    "tiiuae/Falcon3-1B-Base",
    "tiiuae/Falcon3-3B-Base",
    "tiiuae/Falcon3-7B-Base",
    "tiiuae/Falcon3-Mamba-7B-Base",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_falcon_3(record_forge_property, variant):

    if variant == "tiiuae/Falcon3-Mamba-7B-Base" or variant == "tiiuae/Falcon3-7B-Base":
        pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 36 GB)")
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="falcon3", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    tokenizer = AutoTokenizer.from_pretrained(variant)
    model = AutoModelForCausalLM.from_pretrained(variant)
    model.config.use_cache = False
    model.config.return_dict = False

    input_text = "Write a function to calculate the factorial of a number"
    input_data = tokenizer.encode(input_text, return_tensors="pt")

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=input_data, module_name=module_name)

    # Model Verification
    verify([input_data], model, compiled_model)
