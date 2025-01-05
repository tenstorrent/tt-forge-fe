# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import torch
import forge
import os
from test.models.utils import build_module_name


def generate_model_yoloV5I320_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size

    model = download_model(torch.hub.load, variant, name, pretrained=True)

    input_shape = (1, 3, 320, 320)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = ["n", "s", "m", "l", "x"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_320x320(record_forge_property, size):
    variant = "yolov5" + size

    module_name = build_module_name(
        framework="pt", model="yolo_v5", variant=variant, task="imgcls", source="torchhub", suffix="320x320"
    )

    model, inputs, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


def generate_model_yoloV5I640_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)

    input_shape = (1, 3, 640, 640)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = ["n", "s", "m", "l", "x"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_640x640(record_forge_property, size):
    variant = "yolov5" + size

    module_name = build_module_name(
        framework="pt", model="yolo_v5", variant=variant, task="imgcls", source="torchhub", suffix="640x640"
    )

    model, inputs, _ = generate_model_yoloV5I640_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


def generate_model_yoloV5I480_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)
    input_shape = (1, 3, 480, 480)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_480x480(record_forge_property, size):
    variant = "yolov5" + size

    module_name = build_module_name(
        framework="pt", model="yolo_v5", variant=variant, task="imgcls", source="torchhub", suffix="480x480"
    )

    model, inputs, _ = generate_model_yoloV5I480_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_yolov5_1280x1280(record_forge_property):
    variant = "yolov5s"
    module_name = build_module_name(
        framework="pt", model="yolo_v5", variant=variant, task="imgcls", source="torchhub", suffix="1280x1280"
    )

    model = download_model(torch.hub.load, "ultralytics/yolov5", variant, pretrained=True)

    input_shape = (1, 3, 1280, 1280)
    input_tensor = torch.rand(input_shape)
    inputs = [input_tensor]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
