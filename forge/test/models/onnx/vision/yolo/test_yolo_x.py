# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import cv2, torch
import numpy as np
import onnx
import requests

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task


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
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_yolox_onnx(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="yolox",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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
    inputs = [img_tensor]

    # Load and validate the ONNX model
    onnx_model_path = f"third_party/confidential_customer_models/generated/files/{variant}.onnx"
    model_name = f"{variant}_onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
