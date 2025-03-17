# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge, os
import pytest
import cv2, torch
import numpy as np
import onnx
from forge.verify.backend import verify_module
from forge import DepricatedVerifyConfig
from forge.verify.config import TestKind
import requests

# from forge._C.backend_api import BackendDevice


def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img = torch.from_numpy(padded_img)
    return padded_img


variants = ["yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l", "yolox_darknet", "yolox_x"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Not supported")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_yolox_onnx(variant, test_device):

    # forge configuration parameters
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b

    # prepare input
    if variant in ["yolox_nano", "yolox_tiny"]:
        input_shape = (416, 416)
    else:
        input_shape = (640, 640)

    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    response = requests.get(url)
    with open("input.jpg", "wb") as f:
        f.write(response.content)
    img = cv2.imread("input.jpg")
    img_tensor = preprocess(img, input_shape)
    img_tensor = img_tensor.unsqueeze(0)

    # Load and validate the ONNX model
    onnx_model_path = f"third_party/confidential_customer_models/generated/files/{variant}.onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    model_name = f"onnx_{variant}"
    tt_model = forge.OnnxModule(model_name, onnx_model)

    # PCC
    if test_device.arch == BackendDevice.Wormhole_B0:
        if variant == "yolox_nano":
            pcc = 0.93
        else:
            pcc = 0.99
    elif test_device.arch == BackendDevice.Grayskull:
        if variant == "yolox_nano":
            pcc = 0.91
        elif variant in ["yolox_m", "yolox_darknet"]:
            pcc = 0.92
        elif variant in ["yolox_s", "yolox_l"]:
            pcc = 0.93
        elif variant == "yolox_x":
            pcc = 0.94
        else:
            pcc = 0.99

    # Inference
    verify_module(
        tt_model,
        input_shapes=([img_tensor.shape]),
        inputs=([img_tensor]),
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=pcc,
        ),
    )
