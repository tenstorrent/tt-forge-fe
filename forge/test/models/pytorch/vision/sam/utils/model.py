# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import requests
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


def get_model_inputs(variant, input_url="https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"):

    framework_model = SamModel.from_pretrained(variant).to("cpu")
    processor = SamProcessor.from_pretrained(variant)

    img_url = input_url
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    input_points = [[[450, 600]]]

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to("cpu")
    sample_inputs = (
        inputs["pixel_values"],
        inputs["input_points"],
    )
    return framework_model, sample_inputs


class SamWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_points):
        return self.model(pixel_values=pixel_values, input_points=input_points).pred_masks.cpu()
