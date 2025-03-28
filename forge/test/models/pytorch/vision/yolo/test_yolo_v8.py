# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from ultralytics import YOLO

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.utils.yolovx_utils import get_test_input
from test.models.utils import Framework, Source, Task, build_module_name


class YoloV8Wrapper(torch.nn.Module):
    def __init__(self, model_url: str):
        super().__init__()
        self.yolo = YOLO(model_url)

    def forward(self, x):
        return self.yolo.model.forward(x, augment=False)


@pytest.mark.xfail(reason=" ValueError: not enough values to unpack (expected 3, got 0)")
@pytest.mark.nightly
def test_yolo_v8_inference(forge_property_recorder):

    model_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

    # Build Module Name

    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="yolo_v8",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Record Forge property

    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    # Load framework_model and input

    framework_model = YoloV8Wrapper(model_url)
    inputs = [get_test_input()]

    # Compile with Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
