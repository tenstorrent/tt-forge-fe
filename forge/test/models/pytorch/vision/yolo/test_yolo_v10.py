# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import subprocess

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
def test_yolov10(record_forge_property):

    # Upgrading ultralytics to run yolov10 model
    subprocess.run(["pip", "install", "ultralytics==8.3.91"], check=True)

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="Yolov10",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    record_forge_property("group", "priority_1")
    record_forge_property("tags.model_name", module_name)

    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"

    # Load model
    framework_model = YOLO(MODEL_URL)
    framework_model.eval()
    framework_model = YoloWrapper(framework_model)

    # Load input
    dataset = load_dataset("huggingface/cats-image", split="test")
    image = dataset[0]["image"]
    image_np = np.array(image)
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float()

    # Forge compile framework model
    inputs = [image_tensor]
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    # Reverting to ultralytics version in the core_requirements.txt
    subprocess.run(["pip", "install", "--no-cache-dir", "ultralytics==8.0.145"], check=True)
