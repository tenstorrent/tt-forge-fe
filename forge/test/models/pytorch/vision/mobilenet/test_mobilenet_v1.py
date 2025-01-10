# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest

import torch

import forge
from test.utils import download_model
from test.models.pytorch.vision.mobilenet.utils.mobilenet_v1 import MobileNetV1

import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from forge.verify.compare import compare_with_golden


def generate_model_mobilenetV1_base_custom_pytorch(test_device, variant):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()

    # Create Forge module from PyTorch model
    model = MobileNetV1(9)

    input_shape = (1, 3, 64, 64)
    image_tensor = torch.rand(*input_shape)

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.xfail(reason="RuntimeError: Invalid arguments to reshape")
def test_mobilenetv1_basic(test_device):
    model, inputs, _ = generate_model_mobilenetV1_base_custom_pytorch(
        test_device,
        None,
    )

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_mobilenet_v1_basic")
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = model(*inputs)
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


def generate_model_mobilenetv1_imgcls_hf_pytorch(test_device, variant):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
def test_mobilenetv1_192(test_device):
    model, inputs, _ = generate_model_mobilenetv1_imgcls_hf_pytorch(
        test_device,
        "google/mobilenet_v1_0.75_192",
    )
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_mobilenet_v1_192")


def generate_model_mobilenetV1I224_imgcls_hf_pytorch(test_device, variant):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
def test_mobilenetv1_224(test_device):
    model, inputs, _ = generate_model_mobilenetV1I224_imgcls_hf_pytorch(
        test_device,
        "google/mobilenet_v1_1.0_224",
    )
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_mobilenet_v1_224")
