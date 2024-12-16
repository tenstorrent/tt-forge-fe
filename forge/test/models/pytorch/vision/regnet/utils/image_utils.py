# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from test.utils import download_model

import requests
from PIL import Image
from transformers import AutoImageProcessor


def preprocess_input_data(image_url, variant):
    # Load preprocessor
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)

    # Load dataset
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Preprocess the image
    image_tensor = preprocessor(images=image, return_tensors="pt").pixel_values

    return [image_tensor]
