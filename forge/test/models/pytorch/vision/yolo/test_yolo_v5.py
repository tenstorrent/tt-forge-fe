# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

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
    pytest.param(
        "x",
        id="yolov5x",
        marks=[
            pytest.mark.xfail(
                reason="[RuntimeError][Conv2d] bias_ntiles == weight_matrix_width_ntile Issue Link: https://github.com/tenstorrent/tt-mlir/issues/3949"
            )
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size)
def test_yolov5_320x320(restore_package_versions, size):

    pcc = 0.99
    if size == "l":
        pcc = 0.95

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV5,
        variant="yolov5" + size,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="320x320",
    )

    framework_model, inputs, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )


def generate_model_yoloV5I640_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size
    model = fetch_model(name, f"{base_url}/{name}.pt", yolov5_loader, variant=variant)

    input_shape = (1, 3, 640, 640)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = [
    pytest.param("n", id="yolov5n"),
    pytest.param("s", id="yolov5s"),
    pytest.param("m", id="yolov5m"),
    pytest.param("l", id="yolov5l"),
    pytest.param("x", id="yolov5x", marks=[pytest.mark.xfail]),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size)
def test_yolov5_640x640(restore_package_versions, size):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV5,
        variant="yolov5" + size,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="640x640",
    )

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
    model = fetch_model(name, f"{base_url}/{name}.pt", yolov5_loader, variant=variant)
    input_shape = (1, 3, 480, 480)
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
def test_yolov5_480x480(restore_package_versions, size):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV5,
        variant="yolov5" + size,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="480x480",
    )

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
@pytest.mark.xfail
def test_yolov5_1280x1280(restore_package_versions, variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV5,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="1280x1280",
    )

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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
