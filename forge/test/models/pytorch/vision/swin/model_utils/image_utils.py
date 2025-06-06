# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import requests
import torch
from PIL import Image
from torchvision import models


def load_image(image_path, feature_extractor):
    image = Image.open(requests.get(image_path, stream=True).raw)
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    return [img_tensor]


variants_with_weights = {
    "swin_t": "Swin_T_Weights",
    "swin_s": "Swin_S_Weights",
    "swin_b": "Swin_B_Weights",
    "swin_v2_t": "Swin_V2_T_Weights",
    "swin_v2_s": "Swin_V2_S_Weights",
    "swin_v2_b": "Swin_V2_B_Weights",
}


def load_model(variant):
    weight_name = variants_with_weights[variant]
    weights = getattr(models, weight_name).DEFAULT
    model = getattr(models, variant)(weights=weights)
    model.eval()
    return model


def load_input(weight_name):
    weights = getattr(models, weight_name).DEFAULT
    preprocess = weights.transforms()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)
    return [batch_t]
