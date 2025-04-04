# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from datasets import load_dataset
from torchvision import transforms
from ultralytics.nn.tasks import DetectionModel


class YoloWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        result = self.model(image)[0]
        return result[0]


def load_yolo_model_and_image(url):
    # Load YOLO model weights
    weights = torch.hub.load_state_dict_from_url(url, map_location="cpu")

    # Initialize and load model
    model = DetectionModel(cfg=weights["model"].yaml)
    model.load_state_dict(weights["model"].float().state_dict())
    model.eval()

    # Load sample image and preprocess
    dataset = load_dataset("huggingface/cats-image", split="test[:1]")
    image = dataset[0]["image"]
    preprocess = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0)

    return model, image_tensor
