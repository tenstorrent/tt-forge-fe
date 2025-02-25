# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def load_model():
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    model.eval()
    return model


def load_input():
    test_input = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(test_input, stream=True).raw)
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    inputs = image_processor(images=image, return_tensors="pt")
    return [inputs["pixel_values"]]
