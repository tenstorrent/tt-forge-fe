# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, build_module_name
from test.utils import fetch_model, yolov5_loader

base_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0"


def generate_model_yoloV5I320_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size

    model = fetch_model(name, f"{base_url}/{name}.pt", yolov5_loader, variant=variant)

    input_shape = (1, 3, 320, 320)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = [
    pytest.param("n", id="yolov5n"),
    pytest.param("s", id="yolov5s"),
    pytest.param("m", id="yolov5m"),
    pytest.param("l", id="yolov5l"),
    pytest.param("x", id="yolov5x"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size)
def test_yolov5_320x320(forge_property_recorder, size):
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
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, inputs, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def generate_model_yoloV5I640_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size
    model = fetch_model(name, f"{base_url}/{name}.pt", yolov5_loader, variant=variant)

    input_shape = (1, 3, 640, 640)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = [
    pytest.param("n", id="yolov5n"),
    pytest.param(
        "s",
        id="yolov5s",
        marks=[
            pytest.mark.xfail(
                reason="Out of Memory: Not enough space to allocate 73859072 B L1 buffer across 64 banks, where each bank needs to store 1154048 B"
            )
        ],
    ),
    pytest.param("m", id="yolov5m"),
    pytest.param("l", id="yolov5l"),
    pytest.param("x", id="yolov5x"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size)
def test_yolov5_640x640(forge_property_recorder, size):
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
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, inputs, _ = generate_model_yoloV5I640_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def generate_model_yoloV5I480_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size
    model = fetch_model(name, f"{base_url}/{name}.pt", yolov5_loader, variant=variant)
    input_shape = (1, 3, 480, 480)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = [
    pytest.param(
        "n",
        marks=[pytest.mark.xfail],
    ),
    pytest.param("s", id="yolov5s"),
    pytest.param("m", id="yolov5m"),
    pytest.param("l", id="yolov5l"),
    pytest.param("x", id="yolov5x"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size)
def test_yolov5_480x480(forge_property_recorder, size):
    if size != "n":
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
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, inputs, _ = generate_model_yoloV5I480_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["yolov5s"])
def test_yolov5_1280x1280(forge_property_recorder, variant):
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
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model = fetch_model(
        variant,
        f"{base_url}/{variant}.pt",
        yolov5_loader,
        variant="ultralytics/yolov5",
    )

    input_shape = (1, 3, 1280, 1280)
    input_tensor = torch.rand(input_shape)
    inputs = [input_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
