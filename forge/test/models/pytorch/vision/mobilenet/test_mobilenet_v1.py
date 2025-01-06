# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.mobilenet.utils.mobilenet_v1 import MobileNetV1
from test.models.utils import Framework, Source, build_module_name
from test.utils import download_model


def generate_model_mobilenetV1_base_custom_pytorch():
    # Create Forge module from PyTorch model
    model = MobileNetV1(9)

    input_shape = (1, 3, 64, 64)
    image_tensor = torch.rand(*input_shape)

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_mobilenetv1_basic(record_forge_property):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="mobilenet_v1", variant="basic")

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_mobilenetV1_base_custom_pytorch()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
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
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="mobilnet_v1", variant=variant, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_mobilenetv1_imgcls_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
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
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="mobilnet_v1", variant=variant, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_mobilenetV1I224_imgcls_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
