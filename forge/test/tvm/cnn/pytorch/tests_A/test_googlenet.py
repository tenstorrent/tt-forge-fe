# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest

import torch
from torchvision import transforms, models

from forge import (
    PyTorchModule,
    VerifyConfig,
)
from forge.config import CompileDepth, _get_global_compiler_config
from forge.verify.backend import verify_module
from forge.verify.config import TestKind
from test.utils import download_model
import forge


def test_tvm_googlenet(test_kind, test_device):
    if test_device.arch == forge.BackendDevice.Wormhole_B0:
        pytest.skip("Skip for Wormhole_B0")
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    os.environ["FORGE_EXTRA_L1_MARGIN"] = "30000"
    compiler_cfg = _get_global_compiler_config()

    compiler_cfg.balancer_policy = "CNN"
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    pytorch_model = download_model(torch.hub.load, 
        "pytorch/vision:v0.10.0", "googlenet", pretrained=True
    )
    module = PyTorchModule("googlenet", pytorch_model)

    input_shape = (1, 3, 224, 224)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.98,
        ),
    )


def test_googlenet_torchvision(test_kind, test_device):
    
    pytest.skip("Needs padding")
    
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    import os
    # This will allow the test to pass but we should use conv padding to fix the issue instead
    # os.environ["FORGE_EXTRA_L1_MARGIN"] = "30000"
    # unknown padding to add 
    compiler_cfg = _get_global_compiler_config()

    compiler_cfg.balancer_policy = "CNN"

    model = download_model(models.googlenet, pretrained=True)
    module = PyTorchModule("googlenet_pt", model)

    input_shape = (1, 3, 256, 256)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
