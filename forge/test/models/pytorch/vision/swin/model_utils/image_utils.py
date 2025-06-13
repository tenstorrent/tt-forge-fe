# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from torchvision import models


def load_image(feature_extractor):
    dataset = load_dataset("cifar10", split="test")
    image = dataset[0]["img"]
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
