# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import forge
import pytest
from test.models.pytorch.vision.fpn.utils.model import FPNWrapper
from test.models.utils import build_module_name, Framework, Task


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_fpn_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="fpn", source="torchvision")

    record_forge_property("module_name", module_name)

    # Load FPN model
    model = FPNWrapper()

    feat0 = torch.rand(1, 256, 64, 64)
    feat1 = torch.rand(1, 512, 16, 16)
    feat2 = torch.rand(1, 2048, 8, 8)

    inputs = [feat0, feat1, feat2]

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
