# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import timm
import torch
from loguru import logger
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoFeatureExtractor, ResNetForImageClassification

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, build_module_name
from test.utils import download_model


def generate_model_resnet_imgcls_hf_pytorch(variant):
    # Load ResNet feature extractor and model checkpoint from HuggingFace
    model_ckpt = variant
    feature_extractor = download_model(AutoFeatureExtractor.from_pretrained, model_ckpt)
    model = download_model(ResNetForImageClassification.from_pretrained, model_ckpt)

    # Load data sample
    try:
        url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image = torch.rand(1, 3, 256, 256)

    # Data preprocessing
    inputs = feature_extractor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    return model, [pixel_values], {}


@pytest.mark.nightly
def test_resnet(record_forge_property):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="resnet", variant="50", source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_resnet_imgcls_hf_pytorch(
        "microsoft/resnet-50",
    )

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_resnet_imgcls_timm_pytorch(variant):
    # Load ResNet50 feature extractor and model from TIMM
    model = download_model(timm.create_model, variant, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # Load data sample
    try:
        url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image = torch.rand(1, 3, 256, 256)

    # Data preprocessing
    pixel_values = transform(image).unsqueeze(0)

    return model, [pixel_values], {}


@pytest.mark.nightly
def test_resnet_timm(record_forge_property):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="resnet", source=Source.TIMM, variant="50")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_resnet_imgcls_timm_pytorch("resnet50")

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
