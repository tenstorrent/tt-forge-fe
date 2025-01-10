# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import forge
import pytest
from test.models.pytorch.vision.fpn.utils.model import FPNWrapper


@pytest.mark.nightly
def test_fpn_pytorch(test_device):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load FPN model
    model = FPNWrapper()

    feat0 = torch.rand(1, 256, 64, 64)
    feat1 = torch.rand(1, 512, 16, 16)
    feat2 = torch.rand(1, 2048, 8, 8)

    inputs = [feat0, feat1, feat2]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_fpn")
