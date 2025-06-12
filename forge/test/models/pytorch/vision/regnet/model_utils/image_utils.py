# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset
from transformers import AutoImageProcessor

from test.utils import download_model


def preprocess_input_data(variant):
    # Load preprocessor
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)

    dataset = load_dataset("cifar10", split="test")
    image = dataset[0]["img"]

    # Preprocess the image
    image_tensor = preprocessor(images=image, return_tensors="pt").pixel_values

    return [image_tensor]
