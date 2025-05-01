# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/state-spaces/mamba-2.8b-hf

import pytest
import torch
from transformers import AutoTokenizer, MambaForCausalLM

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import DepricatedVerifyConfig, verify

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
    pytest.param("state-spaces/mamba-790m-hf"),
    pytest.param(
        "state-spaces/mamba-2.8b-hf",
        marks=pytest.mark.skip(
            reason="Insufficient host DRAM to run this model (requires a bit more than 29 GB during compile time)"
        ),
    ),
    pytest.param(
        "state-spaces/mamba-1.4b-hf",
        marks=pytest.mark.skip(
            reason="Insufficient host DRAM to run this model (requires a bit more than 29 GB during compile time)"
        ),
    ),
    pytest.param(
        "state-spaces/mamba-370m-hf",
        marks=pytest.mark.skip(
            reason="Insufficient host DRAM to run this model (requires a bit more than 23 GB during compile time)"
        ),
    ),
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_mamba(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="mamba", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(MambaForCausalLM.from_pretrained, variant, use_cache=False)
    model.eval()
    framework_model = Wrapper(model)

    # Prepare input
    prompt = "Hey how are you doing?"
    inputs = [tokenizer(prompt, return_tensors="pt")["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        verify_cfg=DepricatedVerifyConfig(verify_forge_codegen_vs_framework=True),
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
