# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import requests
import pytest
import torch
from PIL import Image
import cv2
import numpy as np
import onnx

from test.models.pytorch.vision.yolo.test_yolo_v5 import generate_model_yoloV5I320_imgcls_torchhub_pytorch
import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task


variants = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]


@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_yolo_v5_320x320_onnx(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="yolo_v5",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="320x320",
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load data sample
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    input_size = 320
    _, pixel_values, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=input_size,
    )
    inputs = [pixel_values]

    # Load onnx model
    onnx_model_path = f"./third_party/confidential_customer_models/generated/files/{variant}_{input_size}.onnx"
    model_name = f"{variant}_{input_size}_onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_yolo_v5_480x480_onnx(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="yolo_v5",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="480x480",
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    input_size = 480

    _, pixel_values, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=input_size,
    )
    inputs = [pixel_values]

    # Load the ONNX model
    onnx_model_path = f"./third_party/confidential_customer_models/generated/files/{variant}_{input_size}.onnx"
    model_name = f"{variant}_{input_size}_onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_yolo_v5_640x640_onnx(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="yolo_v5",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="640x640",
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    input_size = 640
    _, pixel_values, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=input_size,
    )
    inputs = [pixel_values]

    # Load the ONNX model
    onnx_model_path = f"./third_party/confidential_customer_models/generated/files/{variant}_{input_size}.onnx"
    model_name = f"{variant}_{input_size}_onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
