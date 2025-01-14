# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, build_module_name
from test.utils import download_model


def generate_model_yoloV5I320_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size

    model = download_model(torch.hub.load, variant, name, pretrained=True)

    input_shape = (1, 3, 320, 320)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = ["n", "s", "m", "l", "x"]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_320x320(record_forge_property, size):
    if size != "s":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="yolo_v5",
        variant="yolov5" + size,
        task="imgcls",
        source="torchhub",
        suffix="320x320",
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_yoloV5I640_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)

    input_shape = (1, 3, 640, 640)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = ["n", "s", "m", "l", "x"]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_640x640(record_forge_property, size):
    if size != "s":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="yolo_v5",
        variant="yolov5" + size,
        task="imgcls",
        source="torchhub",
        suffix="640x640",
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_yoloV5I640_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_yoloV5I480_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)
    input_shape = (1, 3, 480, 480)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


@pytest.mark.nightly
@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_480x480(record_forge_property, size):
    if size != "s":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="yolo_v5",
        variant="yolov5" + size,
        task="imgcls",
        source="torchhub",
        suffix="480x480",
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_yoloV5I480_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["yolov5s"])
def test_yolov5_1280x1280(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="yolo_v5",
        variant=variant,
        task="imgcls",
        source="torchhub",
        suffix="1280x1280",
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model = download_model(torch.hub.load, "ultralytics/yolov5", variant, pretrained=True)

    input_shape = (1, 3, 1280, 1280)
    input_tensor = torch.rand(input_shape)
    inputs = [input_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
