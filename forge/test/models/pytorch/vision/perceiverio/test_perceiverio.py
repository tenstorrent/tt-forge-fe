# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoImageProcessor,
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationFourier,
    PerceiverForImageClassificationLearned,
)

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify


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
    pytest.param("deepmind/vision-perceiver-conv", id="deepmind/vision-perceiver-conv"),
    pytest.param("deepmind/vision-perceiver-learned", id="deepmind/vision-perceiver-learned"),
    pytest.param(
        "deepmind/vision-perceiver-fourier",
        id="deepmind/vision-perceiver-fourier",
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_perceiverio_for_image_classification_pytorch(forge_property_recorder, variant):
    if variant != "deepmind/vision-perceiver-conv":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="perceiverio",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Sample Image
    pixel_values = get_sample_data(variant)

    # Load the model from HuggingFace
    if variant == "deepmind/vision-perceiver-learned":
        framework_model = PerceiverForImageClassificationLearned.from_pretrained(variant, return_dict=False)

    elif variant == "deepmind/vision-perceiver-conv":
        framework_model = PerceiverForImageClassificationConvProcessing.from_pretrained(variant, return_dict=False)

    elif variant == "deepmind/vision-perceiver-fourier":
        framework_model = PerceiverForImageClassificationFourier.from_pretrained(variant, return_dict=False)

    else:
        logger.info(f"The model {variant} is not supported")

    framework_model.eval()

    inputs = [pixel_values]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
