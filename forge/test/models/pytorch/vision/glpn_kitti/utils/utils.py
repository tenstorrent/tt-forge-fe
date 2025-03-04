# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import requests
from PIL import Image
from transformers import GLPNForDepthEstimation, GLPNImageProcessor


def load_model(variant):
    model = GLPNForDepthEstimation.from_pretrained(variant)
    model.eval()
    return model


def load_input(variant):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processor = GLPNImageProcessor.from_pretrained(variant)
    inputs = processor(images=image, return_tensors="pt")
    return [inputs["pixel_values"]]
