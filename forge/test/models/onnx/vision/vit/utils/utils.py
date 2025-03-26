# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoImageProcessor, ViTForImageClassification
from test.utils import download_model
import urllib
from PIL import Image


def load_model(variant):
    model = download_model(ViTForImageClassification.from_pretrained, variant)
    return model


def load_inputs(url, filename, variant):
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)

    image_processor = download_model(AutoImageProcessor.from_pretrained, variant)
    img_tensor = image_processor(input_image, return_tensors="pt").pixel_values
    return [img_tensor]
