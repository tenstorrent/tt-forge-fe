# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.fpn.utils.model import FPNWrapper


@pytest.mark.nightly
def test_fpn_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="fpn", source=Source.TORCHVISION, task=Task.IMAGE_CLASSIFICATION
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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
