# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FalconForCausalLM

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.models_utils import generate_no_cache, pad_inputs
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["tiiuae/falcon-7b-instruct"])
def test_falcon(forge_property_recorder, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 32 GB)")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="falcon", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


variants = [
    pytest.param("tiiuae/Falcon3-1B-Base", marks=pytest.mark.push),
    pytest.param(
        "tiiuae/Falcon3-3B-Base",
        marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 25 GB)"),
    ),
    pytest.param(
        "tiiuae/Falcon3-7B-Base",
        marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 36 GB)"),
    ),
    pytest.param(
        "tiiuae/Falcon3-10B-Base",
        marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB)"),
    ),
    pytest.param(
        "tiiuae/Falcon3-Mamba-7B-Base",
        marks=pytest.mark.skip(reason="Insufficient host DRAM to run this model (requires a bit more than 36 GB)"),
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_falcon_3(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="falcon3", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    if variant in [
        "tiiuae/Falcon3-1B-Base",
        "tiiuae/Falcon3-3B-Base",
        "tiiuae/Falcon3-7B-Base",
        "tiiuae/Falcon3-10B-Base",
    ]:
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")

    # Load model and tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AutoModelForCausalLM.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()

    # prepare input
    input_text = "Write a function to calculate the factorial of a number"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    padded_inputs, seq_len = pad_inputs(inputs)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # post processing
    generated_text = generate_no_cache(
        max_new_tokens=50, model=compiled_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=tokenizer
    )

    print("generated_text : ", generated_text)
