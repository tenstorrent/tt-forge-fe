# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# CodeGen Demo - CasualLM

import os
import torch
import pytest
from test.utils import download_model
from transformers import AutoTokenizer, CodeGenForCausalLM

import forge
from forge import VerifyConfig
from forge.verify.config import TestKind, NebulaGalaxy
from forge.verify.backend import verify_module
from forge._C.backend_api import BackendDevice, BackendType

variants = [
    "Salesforce/codegen-350M-mono",
    # "Salesforce/codegen-350M-multi", # Currently not supported
    # "Salesforce/codegen-350M-nl", # Currently not supported
]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_codegen(test_device, variant):
    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{32*1024}"
    pcc = 0.98
    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.default_dram_parameters = False
        compiler_cfg.balancer_policy = "Ribbon"
        pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.98
    # DRAM stream limit
    compiler_cfg.balancer_op_override("matmul_1829", "grid_shape", (2, 8))

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

    forge_model = forge.PyTorchModule("pt_codegen", framework_model)
    verify_module(
        forge_model,
        input_shapes=[
            (
                input_ids.shape,
                attn_mask.shape,
            )
        ],
        inputs=[
            (
                input_ids,
                attn_mask,
            )
        ],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids
            if "FORGE_NEB_GALAXY_CI" in os.environ and int(os.environ.get("FORGE_NEB_GALAXY_CI")) == 1
            else [0],
            pcc=pcc,
        ),
    )
