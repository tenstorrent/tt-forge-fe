# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
from transformers import (
    AutoImageProcessor,
    SegformerForImageClassification,
    SegformerConfig,
)

import requests
import pytest
from PIL import Image


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


variants_img_classification = [
    "nvidia/mit-b0",
    "nvidia/mit-b1",
    "nvidia/mit-b2",
    "nvidia/mit-b3",
    "nvidia/mit-b4",
    "nvidia/mit-b5",
]


@pytest.mark.parametrize("variant", variants_img_classification)
def test_segformer_image_classification_pytorch(test_device, variant):

    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Set model configurations
    config = SegformerConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = SegformerConfig(**config_dict)

    # Load the model from HuggingFace
    model = SegformerForImageClassification.from_pretrained(variant, config=config)
    model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)

    compiled_model = forge.compile(
        model, sample_inputs=[pixel_values], module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )
