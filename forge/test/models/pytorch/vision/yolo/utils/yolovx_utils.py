# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from datasets import load_dataset
from torch.hub import load_state_dict_from_url
from torchvision import transforms
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils.torch_utils import smart_inference_mode


def get_test_input():
    dataset = load_dataset("huggingface/cats-image", split="test[:1]")
    image = dataset[0]["image"]
    preprocess = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor


class WorldModelWrapper(torch.nn.Module):
    def __init__(self, model_url: str):
        super().__init__()
        ckpt = load_state_dict_from_url(model_url, map_location="cpu")
        cfg_path = ckpt.get("cfg", "yolov8s-world.yaml")
        model = WorldModel(cfg=cfg_path)

        if "model" in ckpt:
            state_dict = ckpt["model"].float().state_dict()
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.model = model

    @smart_inference_mode()
    def forward(self, x):
        return self.model(x)[0][0]
