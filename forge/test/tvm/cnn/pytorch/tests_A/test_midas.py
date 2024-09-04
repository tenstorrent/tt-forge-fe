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


def test_MIDAS_pytorch(
    test_kind,
    test_device,
):
    pytest.skip()  # Takes too long post commit

    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS

    model = download_model(torch.hub.load, "intel-isl/MiDaS", "MiDaS_small")
    module = PyTorchModule("MIDAS_torch", model)

    input_shape = (1, 3, 384, 384)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
