# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch
from datasets import load_dataset
from ultralytics import YOLO

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name


class YoloWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        results = self.model.predict(image, conf=0.25, verbose=False)[0]
        boxes = results.boxes.xyxy
        class_ids = results.boxes.cls
        confidences = results.boxes.conf
        return boxes, class_ids, confidences


@pytest.mark.xfail(reason="TypeError: type Tensor doesn't define __round__ method")
@pytest.mark.nightly
def test_yolov10(forge_property_recorder):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="Yolov10",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"

    # Load model
    framework_model = YOLO(MODEL_URL)
    framework_model.eval()
    framework_model = YoloWrapper(framework_model)

    # Load input
    dataset = load_dataset("huggingface/cats-image", split="test[:1]")
    image = dataset[0]["image"]
    image_np = np.array(image)
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float()

    # Forge compile framework model
    inputs = [image_tensor]
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
