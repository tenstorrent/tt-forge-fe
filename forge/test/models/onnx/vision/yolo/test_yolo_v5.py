# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import requests
import pytest
import torch
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from yolov5.utils.dataloaders import exif_transpose, letterbox
import onnx, pytest

# TODO: These are old forge, we should update them to the currently version.
# import forge
# from forge.verify.backend import verify_module
# from forge import DepricatedVerifyConfig
# from forge.verify.config import TestKind
# from forge._C.backend_api import BackendDevice


def data_preprocessing(ims: Image.Image, size: tuple) -> tuple:
    """Data preprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : Image.Image
        Input image
    size : tuple
        Desired image size

    Returns
    -------
    tuple
        List of images, number of samples, filenames, image size, inference size, preprocessed images
    """

    _, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
    shape0, shape1, files = [], [], []  # image and inference shapes, filenames

    for i, im in enumerate(ims):
        f = f"image{i}"  # filename
        im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
        files.append(Path(f).with_suffix(".jpg").name)
        if im.shape[0] < 5:  # image in CHW
            im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
        im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
        s = im.shape[:2]  # HWC
        shape0.append(s)  # image shape
        g = max(size) / max(s)  # gain
        shape1.append([int(y * g) for y in s])
        ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
    shape1 = [size[0] for _ in np.array(shape1).max(0)]  # inf shape
    x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
    x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
    x = torch.from_numpy(x) / 255  # uint8 to fp16/32
    return x


variants = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="CCM is not public yet.")
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
@pytest.mark.skip(reason="CCM is not public yet.")
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
@pytest.mark.skip(reason="CCM is not public yet.")
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
