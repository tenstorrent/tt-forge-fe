# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from test.utils import download_model
import torch
from PIL import Image
from torchvision import transforms
from loguru import logger
import pytest
import forge
from pytorchcv.model_provider import get_model as ptcv_get_model
import os
from test.models.utils import build_module_name, Framework


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_alexnet_torchhub(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="alexnet", source="torchhub")

    record_forge_property("module_name", module_name)

    # Load model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "alexnet", pretrained=True)
    framework_model.eval()

    # Load and pre-process image
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(input_image).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    inputs = [img_tensor]
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_alexnet_osmr(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="alexnet", source="osmr")

    record_forge_property("module_name", module_name)

    # Load model
    framework_model = download_model(ptcv_get_model, "alexnet", pretrained=True)
    framework_model.eval()

    # Load and pre-process image
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(input_image).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    inputs = [img_tensor]
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)
