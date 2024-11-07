# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
import urllib
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import requests
from loguru import logger
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import MobileNetV2ForSemanticSegmentation
import os


def generate_model_mobilenetV2_imgcls_torchhub_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
def test_mobilenetv2_basic(test_device):
    model, inputs, _ = generate_model_mobilenetV2_imgcls_torchhub_pytorch(
        test_device,
        "pytorch/vision:v0.10.0",
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name="mobilenetv2_basic")


def generate_model_mobilenetV2I96_imgcls_hf_pytorch(test_device, variant):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    image_tensor = inputs.pixel_values

    return model, [image_tensor], {}


@pytest.mark.nightly
def test_mobilenetv2_96(test_device):
    model, inputs, _ = generate_model_mobilenetV2I96_imgcls_hf_pytorch(
        test_device,
        "google/mobilenet_v2_0.35_96",
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name="mobilenetv2_96")


def generate_model_mobilenetV2I160_imgcls_hf_pytorch(test_device, variant):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    image_tensor = inputs.pixel_values

    return model, [image_tensor], {}


@pytest.mark.nightly
def test_mobilenetv2_160(test_device):
    model, inputs, _ = generate_model_mobilenetV2I160_imgcls_hf_pytorch(
        test_device,
        "google/mobilenet_v2_0.75_160",
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name="mobilenetv2_160")


def generate_model_mobilenetV2I244_imgcls_hf_pytorch(test_device, variant):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

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
def test_mobilenetv2_224(test_device):
    model, inputs, _ = generate_model_mobilenetV2I244_imgcls_hf_pytorch(
        test_device,
        "google/mobilenet_v2_1.0_224",
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name="mobilenetv2_224")


def generate_model_mobilenetV2_imgcls_timm_pytorch(test_device, variant):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
def test_mobilenetv2_timm(test_device):
    model, inputs, _ = generate_model_mobilenetV2_imgcls_timm_pytorch(
        test_device,
        "mobilenetv2_100",
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name="mobilenetv2_timm")


def generate_model_mobilenetV2_semseg_hf_pytorch(test_device, variant):
    # This variant with input size 3x224x224 works with manual kernel fracturing
    # of the first op. Pad between input activations and first convolution needs
    # to be hoist to the input in order for pre-striding to work (no need for
    # manual kernel fracturing).

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

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


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_mobilenetv2_deeplabv3(variant, test_device):
    model, inputs, _ = generate_model_mobilenetV2_semseg_hf_pytorch(
        test_device,
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name="mobilenetv2_deeplabv3")
