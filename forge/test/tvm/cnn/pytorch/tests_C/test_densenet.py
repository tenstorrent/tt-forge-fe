# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import torch
from torchvision import models

from forge import (
    PyTorchModule,
    VerifyConfig,
)
from forge.config import CompileDepth, _get_global_compiler_config
from forge.verify.backend import verify_module
from forge.verify.config import TestKind
from test.utils import download_model


def test_densenet_121(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    import os
    os.environ["FORGE_DISABLE_CONSTANT_FOLDING"] = "1"

    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet121", pretrained=True)
    module = PyTorchModule("densenet121_pt", model)

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

    os.environ.pop('FORGE_DISABLE_CONSTANT_FOLDING', None)
    
def test_densenet_169(test_kind, test_device):
    
    pytest.skip("Timeouts if all tests are run")

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    import os
    os.environ["FORGE_DISABLE_CONSTANT_FOLDING"] = "1"
    os.environ["FORGE_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"


    model = download_model(models.densenet169, pretrained=True)
    module = PyTorchModule("densenet169_pt", model)

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

    os.environ.pop('FORGE_DISABLE_CONSTANT_FOLDING', None)

    
def test_densenet_201(test_kind, test_device):
    
    pytest.skip("Timeouts if all tests are run")

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    import os
    os.environ["FORGE_DISABLE_CONSTANT_FOLDING"] = "1"
    os.environ["FORGE_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"


    model = download_model(models.densenet201, pretrained=True)
    module = PyTorchModule("densenet201_pt", model)

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

    os.environ.pop('FORGE_DISABLE_CONSTANT_FOLDING', None)
    
    
def test_densenet_161(test_kind, test_device):
    
    pytest.skip("Timeouts if all tests are run")

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    import os
    os.environ["FORGE_DISABLE_CONSTANT_FOLDING"] = "1"
    os.environ["FORGE_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"


    model = download_model(models.densenet161, pretrained=True)
    module = PyTorchModule("densenet161_pt", model)

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

    os.environ.pop('FORGE_DISABLE_CONSTANT_FOLDING', None)
