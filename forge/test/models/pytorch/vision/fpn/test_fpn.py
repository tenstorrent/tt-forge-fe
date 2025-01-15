# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.fpn.utils.model import FPNWrapper
from test.models.utils import Framework, Source, build_module_name


@pytest.mark.nightly
def test_fpn_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="fpn", source=Source.TORCHVISION)

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load FPN model
    framework_model = FPNWrapper()

    feat0 = torch.rand(1, 256, 64, 64)
    feat1 = torch.rand(1, 512, 16, 16)
    feat2 = torch.rand(1, 2048, 8, 8)

    inputs = [feat0, feat1, feat2]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
