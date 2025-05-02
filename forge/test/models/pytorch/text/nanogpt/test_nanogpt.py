# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify


# Wrapper to get around attention mask
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, None, attention_mask)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "FinancialSupport/NanoGPT",
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_nanogpt_text_generation(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="nanogpt",
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(variant)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(variant, ignore_mismatched_sizes=True, use_cache=False, return_dict=False)

    # Input prompt
    input_prompt = "The financial market showed signs of volatility"

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=150,
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    inputs = [input_ids, attn_mask]

    framework_model = Wrapper(model)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
