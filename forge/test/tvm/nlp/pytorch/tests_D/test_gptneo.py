# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# GPT Neo basic bring-up tests of tracing functionality
#
from forge._C.backend_api import BackendDevice
import pytest

import torch
from transformers import GPTNeoModel, GPTNeoConfig
import os

from forge import (
    PyTorchModule,
    CompileDepth,
    VerifyConfig,
    BackendType,
)
from test.tvm.utils import evaluate_framework_vs_forge
from forge.config import CompileDepth, _get_global_compiler_config
from forge.verify import verify_module
from forge.verify.config import TestKind


def test_gptneo_block(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.FORGE_GRAPH_PRE_PLACER

    torch.manual_seed(52)
    input_shape = (1, 64, 2560)
    config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B", torchscript=True)
    config.num_layers = 1  # For faster model loading
    model = GPTNeoModel(config)
    submodel = model.h[0]
    mod = PyTorchModule("gptneo_block", submodel)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
        uniform_inputs=True,
    )

def test_gptneo_full(test_kind, test_device):

    # Pipegen error on silicon if enabled
    os.environ["FORGE_DISABLE_STABLE_SOFTMAX"] = "1"
    os.environ["FORGE_EXTRA_L1_MARGIN"] = "100000"
    
    if test_kind == TestKind.TRAINING:
        pytest.skip()
    
    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FORGE_GRAPH_PRE_PLACER 

    #Fusing disabled due to tenstorrent/forge#789
    if test_kind == TestKind.INFERENCE and test_device.arch == BackendDevice.Wormhole_B0:
        compiler_cfg.enable_auto_fusing=False

    torch.manual_seed(52)
    input_shape = (1, 256)
    inputs = [torch.randint(0, input_shape[-1], input_shape)]
    config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B", torchscript=True)
    config.num_layers = 1  # For faster model loading
    model = GPTNeoModel(config)
    mod = PyTorchModule("gptneo_full", model)

    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99
    verify_module(
        mod,
        (input_shape,),
        inputs=[inputs],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=pcc,
        ),
        uniform_inputs=True,
    )

    os.environ["FORGE_EXTRA_L1_MARGIN"] = "0"
