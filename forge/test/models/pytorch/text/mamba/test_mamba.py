# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/state-spaces/mamba-2.8b-hf

import pytest
import torch
from transformers import AutoTokenizer, MambaForCausalLM

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
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
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 24 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "state-spaces/mamba-1.4b-hf",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 24 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "state-spaces/mamba-370m-hf",
    ),
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_mamba(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MAMBA,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

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
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
