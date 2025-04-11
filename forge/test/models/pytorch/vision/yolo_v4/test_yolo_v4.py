# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from utils.models.models import *

import forge

from test.models.utils import Framework, Task, build_module_name

variants = ["yolo_v4"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolo_v4_pytorch(forge_property_recorder, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="yolo_v4", variant=variant, task=Task.OBJECT_DETECTION, source="custom"
    )

    cfg = "forge/test/models/pytorch/vision/yolo_v4/utils/yolov4.cfg"

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_model_name(module_name)

    # Load model
    framework_model = Darknet(cfg)
    load_darknet_weights(framework_model, weights[0])
    inputs = torch.rand(1, 3, 480, 640)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
