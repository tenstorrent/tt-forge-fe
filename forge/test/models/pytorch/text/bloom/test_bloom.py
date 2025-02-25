# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


# Wrapper to get around past key values
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, None, attention_mask)
        return output


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["bigscience/bloom-1b1"])
def test_bloom(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="bloom",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CAUSAL_LM,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
