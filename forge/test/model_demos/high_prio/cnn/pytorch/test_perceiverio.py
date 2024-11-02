# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import torch
import requests
from PIL import Image
import pytest
from loguru import logger
from transformers import (
    AutoImageProcessor,
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationLearned,
    PerceiverForImageClassificationFourier,
)
import os


def get_sample_data(model_name):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        height = image_processor.to_dict()["size"]["height"]
        width = image_processor.to_dict()["size"]["width"]
        pixel_values = torch.rand(1, 3, height, width).to(torch.float32)
    return pixel_values


variants = [
    "deepmind/vision-perceiver-conv",
    "deepmind/vision-perceiver-learned",
    "deepmind/vision-perceiver-fourier",
]


@pytest.mark.parametrize("variant", variants)
def test_perceiverio_for_image_classification_pytorch(test_device, variant):

    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Sample Image
    pixel_values = get_sample_data(variant)

    # Load the model from HuggingFace
    if variant == "deepmind/vision-perceiver-learned":
        model = PerceiverForImageClassificationLearned.from_pretrained(variant)

    elif variant == "deepmind/vision-perceiver-conv":
        model = PerceiverForImageClassificationConvProcessing.from_pretrained(variant)

    elif variant == "deepmind/vision-perceiver-fourier":
        model = PerceiverForImageClassificationFourier.from_pretrained(variant)

    else:
        logger.info(f"The model {variant} is not supported")

    model.eval()
    # Run inference on Tenstorrent device
    compiled_model = forge.compile(
        model, sample_inputs=[pixel_values], module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )
