# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

from forge import (
    PyTorchModule,
    VerifyConfig,
)
from forge.config import CompileDepth, _get_global_compiler_config
from forge.verify.backend import verify_module
from forge.verify.config import TestKind
from test.utils import download_model


def test_shufflenetv2_pytorch(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.FORGE_GRAPH_PRE_PLACER

    model = download_model(torch.hub.load, 
        "pytorch/vision:v0.10.0", "shufflenet_v2_x1_0", pretrained=True
    )
    module = PyTorchModule("shufflenet_v2_x1_0", model)

    input_shape = (1, 3, 224, 224)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
