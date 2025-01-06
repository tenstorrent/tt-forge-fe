# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

import forge
from test.utils import download_model
from test.models.pytorch.vision.mobilenet.utils.mobilenet_v1 import MobileNetV1

import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from test.models.utils import build_module_name, Framework


def generate_model_mobilenetV1_base_custom_pytorch():
    # Create Forge module from PyTorch model
    model = MobileNetV1(9)

    input_shape = (1, 3, 64, 64)
    image_tensor = torch.rand(*input_shape)

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_mobilenetv1_basic(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="mobilenet_v1", variant="basic")

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_mobilenetV1_base_custom_pytorch()

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


def generate_model_mobilenetv1_imgcls_hf_pytorch(variant):
    # Create Forge module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)
    # tt_model = forge.PyTorchModule("mobilenet_v1__hf_075_192", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")

    image_tensor = inputs.pixel_values

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_mobilenetv1_192(record_forge_property):
    variant = "google/mobilenet_v1_0.75_192"
    module_name = build_module_name(framework=Framework.PYTORCH, model="mobilnet_v1", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_mobilenetv1_imgcls_hf_pytorch(variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


def generate_model_mobilenetV1I224_imgcls_hf_pytorch(variant):
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
def test_mobilenetv1_224(record_forge_property):
    variant = "google/mobilenet_v1_1.0_224"
    module_name = build_module_name(framework=Framework.PYTORCH, model="mobilnet_v1", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_mobilenetV1I224_imgcls_hf_pytorch(variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)
