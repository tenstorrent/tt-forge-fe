# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomModel

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.models_utils import (
    _prepare_4d_causal_attention_mask_with_cache_position,
)
from test.utils import download_model

BloomModel._prepare_4d_causal_attention_mask_with_cache_position = _prepare_4d_causal_attention_mask_with_cache_position


# Wrapper to get around past key values
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, None, attention_mask)
        return output


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "bigscience/bloom-1b1",
            marks=[pytest.mark.xfail],
        ),
    ],
)
def test_bloom(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="bloom",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CAUSAL_LM,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant, padding_side="left")
    model = download_model(AutoModelForCausalLM.from_pretrained, variant, use_cache=False, return_dict=False)
    model.eval()
    framework_model = Wrapper(model)

    # Prepare input
    test_input = "This is a sample text from "
    input_tokens = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
