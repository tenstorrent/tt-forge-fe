# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from urllib.request import urlopen

import requests
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


def load_timm_model_and_input(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    img = Image.open(
        urlopen("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png")
    )
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    input_batch = transforms(img).unsqueeze(0)
    return model, input_batch


def load_vision_model_and_input(variant, task, weight_name):
    if task == "detection":
        weights = getattr(models.detection, weight_name).DEFAULT
        model = getattr(models.detection, variant)(weights=weights)
    else:
        weights = getattr(models, weight_name).DEFAULT
        model = getattr(models, variant)(weights=weights)

    model.eval()

    # Preprocess image
    preprocess = weights.transforms()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)

    return model, [batch_t]
