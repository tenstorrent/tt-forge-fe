# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from transformers import AutoImageProcessor

from test.utils import download_model


def preprocess_input_data(variant):
    # Load preprocessor
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)

    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image))

    # Preprocess the image
    image_tensor = preprocessor(images=image, return_tensors="pt").pixel_values

    return [image_tensor]
