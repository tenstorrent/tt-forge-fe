# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.fpn.utils.model import FPNWrapper
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
def test_fpn_pytorch(forge_property_recorder):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="fpn", source=Source.TORCHVISION, task=Task.IMAGE_CLASSIFICATION
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load FPN model
    framework_model = FPNWrapper()

    feat0 = torch.rand(1, 256, 64, 64)
    feat1 = torch.rand(1, 512, 16, 16)
    feat2 = torch.rand(1, 2048, 8, 8)

    inputs = [feat0, feat1, feat2]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
