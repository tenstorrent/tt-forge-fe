# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import requests
from PIL import Image


def load_image(image_path, feature_extractor):
    image = Image.open(requests.get(image_path, stream=True).raw)
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    return [img_tensor]
