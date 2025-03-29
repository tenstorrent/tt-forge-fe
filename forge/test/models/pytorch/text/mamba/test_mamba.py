# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/state-spaces/mamba-2.8b-hf

import pytest
import torch
from transformers import AutoTokenizer, MambaForCausalLM

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


# Wrapper to return only the output tensor, excluding cache or additional outputs
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        output = self.model(input_ids)
        return output[0]


variants = [
    pytest.param(
        "state-spaces/mamba-790m-hf",
        marks=[pytest.mark.xfail],
    ),
    "state-spaces/mamba-2.8b-hf",
    "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-370m-hf",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mamba(forge_property_recorder, variant):
    if variant != "state-spaces/mamba-790m-hf":
        pytest.skip("Skipping this variant; only testing the base model (mamba-790m-hf) for now.")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="mamba", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(MambaForCausalLM.from_pretrained, variant)
    model.eval()
    framework_model = Wrapper(model)

    # Prepare input
    prompt = "Hey how are you doing?"
    inputs = [tokenizer(prompt, return_tensors="pt")["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
