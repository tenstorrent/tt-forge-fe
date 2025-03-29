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

# TODO: These are old forge, we should update them to the currently version.
# import forge
# from forge.verify.backend import verify_module
# from forge import DepricatedVerifyConfig
# from forge.verify.config import TestKind
# from forge._C.backend_api import BackendDevice


variants = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Requires restructuring")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_yolo_v5_320x320_onnx(test_device, variant):

    # forge configuration parameters
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b

    input_size = 320

    # Load the ONNX model
    onnx_model_path = f"./third_party/confidential_customer_models/generated/files/{variant}_{input_size}.onnx"
    onnx_model = onnx.load(onnx_model_path)
    model_name = f"{variant}_{input_size}_onnx"

    # Load data sample
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Data preprocessing on Host
    pixel_values = data_preprocessing(image, size=(input_size, input_size))

    # Run inference on Tenstorrent device
    verify_module(
        forge.OnnxModule(model_name, onnx_model),
        input_shapes=([pixel_values.shape]),
        inputs=([pixel_values]),
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


variants = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Requires restructuring")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_yolo_v5_480x480_onnx(test_device, variant):

    # forge configuration parameters
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b
    compiler_cfg.enable_tm_cpu_fallback = True

    input_size = 480

    # Load the ONNX model
    onnx_model_path = f"./third_party/confidential_customer_models/generated/files/{variant}_{input_size}.onnx"
    onnx_model = onnx.load(onnx_model_path)
    model_name = f"{variant}_{input_size}_onnx"

    # Load data sample
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Data preprocessing on Host
    pixel_values = data_preprocessing(image, size=(input_size, input_size))

    # Run inference on Tenstorrent device
    verify_module(
        forge.OnnxModule(model_name, onnx_model),
        input_shapes=([pixel_values.shape]),
        inputs=([pixel_values]),
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


variants = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Requires restructuring")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_yolo_v5_640x640_onnx(test_device, variant):

    # forge configuration parameters
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b

    input_size = 640

    # Load the ONNX model
    onnx_model_path = f"./third_party/confidential_customer_models/generated/files/{variant}_{input_size}.onnx"
    onnx_model = onnx.load(onnx_model_path)
    model_name = f"{variant}_{input_size}_onnx"

    # Load data sample
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Data preprocessing on Host
    pixel_values = data_preprocessing(image, size=(input_size, input_size))

    # Run inference on Tenstorrent device
    verify_module(
        forge.OnnxModule(model_name, onnx_model),
        input_shapes=([pixel_values.shape]),
        inputs=([pixel_values]),
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
