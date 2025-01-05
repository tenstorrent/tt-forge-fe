# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import urllib
import requests
from PIL import Image
from loguru import logger

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import MobileNetV2ForSemanticSegmentation

import forge
from test.utils import download_model
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name


def generate_model_mobilenetV2_imgcls_torchhub_pytorch(variant):
    model = download_model(torch.hub.load, variant, "mobilenet_v2", pretrained=True)

    # Image preprocessing
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # TODO : Choose image preprocessor from torchvision,
    # to make a compatible postprocessing of the predicted class
    preprocessor = download_model(AutoImageProcessor.from_pretrained, "google/mobilenet_v2_1.0_224")
    image_tensor = preprocessor(images=image, return_tensors="pt").pixel_values

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_mobilenetv2_basic(record_forge_property):
    variant = "pytorch/vision:v0.10.0"
    module_name = build_module_name(framework="pt", model="mobilenet_v2_basic", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_mobilenetV2_imgcls_torchhub_pytorch(variant)
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)


def generate_model_mobilenetV2I96_imgcls_hf_pytorch(variant):
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    image_tensor = inputs.pixel_values

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_mobilenetv2_96(record_forge_property):
    variant = "google/mobilenet_v2_0.35_96"
    module_name = build_module_name(framework="pt", model="mobilenetv2", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_mobilenetV2I96_imgcls_hf_pytorch(variant)
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)


def generate_model_mobilenetV2I160_imgcls_hf_pytorch(variant):
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    image_tensor = inputs.pixel_values

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_mobilenetv2_160(record_forge_property):
    variant = "google/mobilenet_v2_0.75_160"
    module_name = build_module_name(framework="pt", model="mobilenet_v2", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_mobilenetV2I160_imgcls_hf_pytorch(
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)


def generate_model_mobilenetV2I244_imgcls_hf_pytorch(variant):
    # Create Forge module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")

    image_tensor = inputs.pixel_values

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_mobilenetv2_224(record_forge_property):
    variant = "google/mobilenet_v2_1.0_224"
    module_name = build_module_name(framework="pt", model="mobilenet_v2", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_mobilenetV2I244_imgcls_hf_pytorch(
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)


def generate_model_mobilenetV2_imgcls_timm_pytorch(variant):
    model = download_model(timm.create_model, variant, pretrained=True)
    # tt_model = forge.PyTorchModule("mobilenet_v2__hf_timm", model)

    # Image load and pre-processing into pixel_values
    try:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        image_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image_tensor = torch.rand(1, 3, 224, 224)

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_mobilenetv2_timm(record_forge_property):
    variant = "mobilenetv2_100"
    module_name = build_module_name(framework="pt", model="mobilenet_v2", variant=variant, source="timm")

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_mobilenetV2_imgcls_timm_pytorch(variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


def generate_model_mobilenetV2_semseg_hf_pytorch(variant):
    # This variant with input size 3x224x224 works with manual kernel fracturing
    # of the first op. Pad between input activations and first convolution needs
    # to be hoist to the input in order for pre-striding to work (no need for
    # manual kernel fracturing).

    # Load model
    framework_model = download_model(MobileNetV2ForSemanticSegmentation.from_pretrained, variant)

    try:
        config = resolve_data_config({}, model=framework_model)
        transform = create_transform(**config)
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    return framework_model, [img_tensor], {}


variants = ["google/deeplabv3_mobilenet_v2_1.0_513"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_mobilenetv2_deeplabv3(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="mobilnet_v2", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_mobilenetV2_semseg_hf_pytorch(
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)
