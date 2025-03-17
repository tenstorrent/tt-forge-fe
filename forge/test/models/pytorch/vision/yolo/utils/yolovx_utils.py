# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import subprocess

import numpy as np
import torch
from datasets import load_dataset
from ultralytics import YOLO, YOLOWorld


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


def install_ultralytics(version):
    """Installs the required version of ultralytics."""
    subprocess.run(["pip", "install", "--no-cache-dir", f"ultralytics=={version}"], check=True)


def load_yolo_model(model_name,model_url):
    """Loads a YOLO model from the given URL."""
    if model_name == "Yolo":
        framework_model = YOLO(model_url)
    else:
        framework_model = YOLOWorld(model_url)
    return YoloWrapper(framework_model)


def get_test_input():
    """Loads and preprocesses the test image."""
    dataset = load_dataset("huggingface/cats-image", split="test")
    image = dataset[0]["image"]
    image_np = np.array(image)
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float()
    return image_tensor
