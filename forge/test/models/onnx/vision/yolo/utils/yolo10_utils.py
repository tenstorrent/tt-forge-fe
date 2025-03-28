# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import transforms
from datasets import load_dataset
from ultralytics.nn.tasks import DetectionModel


class YoloV10Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        return self.model(image)


def load_yolov10_model_and_image():
    # Load YOLOv10n model weights
    weights = torch.hub.load_state_dict_from_url(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt", map_location="cpu"
    )

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
