# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from transformers import AutoImageProcessor


def get_sample_data(model_name):
    dataset = load_dataset("cifar10", split="test")
    image = dataset[0]["img"]

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values
