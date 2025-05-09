# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# CodeGen Demo - CasualLM

import pytest
import torch
from transformers import AutoTokenizer, CodeGenForCausalLM, CodeGenModel

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.models_utils import (
    _prepare_4d_causal_attention_mask_with_cache_position,
)
from test.utils import download_model

CodeGenModel._prepare_4d_causal_attention_mask_with_cache_position = (
    _prepare_4d_causal_attention_mask_with_cache_position
)


variants = [
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-350M-multi",
    "Salesforce/codegen-350M-nl",
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_codegen(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="codegen", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load model (with tokenizer)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    framework_model = download_model(CodeGenForCausalLM.from_pretrained, variant, use_cache=False, return_dict=False)

    # Input prompt
    input_prompt = "def hello_world():"

    # Tokenize input
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    # Wrapper to get around attention mask
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    framework_model = Wrapper(framework_model)

    # Sanity run
    attn_mask = attn_mask.to(torch.float32)

    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
