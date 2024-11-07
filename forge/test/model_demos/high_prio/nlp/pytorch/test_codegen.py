# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# CodeGen Demo - CasualLM

import torch
import pytest
from test.utils import download_model
from transformers import AutoTokenizer, CodeGenForCausalLM

import forge

variants = [
    "Salesforce/codegen-350M-mono",
    # "Salesforce/codegen-350M-multi", # Currently not supported
    # "Salesforce/codegen-350M-nl", # Currently not supported
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_codegen(test_device, variant):
    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

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
    input_ids = input_ids.to(torch.int32)
    attn_mask = attn_mask.to(torch.float32)
    out = framework_model(input_ids, attn_mask)

    inputs = [input_ids, attn_mask]
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )
