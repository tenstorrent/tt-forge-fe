# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os

from PIL import Image
import numpy as np

import onnx
import torch

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task


########
# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype="float32")
    image_data /= 255.0
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


#########


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.nightly
def test_yolov3_tiny_onnx(forge_property_recorder):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="yolov_3",
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Image preprocessing
    pil_img = Image.open("third_party/confidential_customer_models/model_2/onnx/saved/yolo_v3/carvana.jpg")
    # input
    image_data = preprocess(pil_img)
    image_size = np.array([pil_img.size[1], pil_img.size[0]], dtype=np.int32).reshape(1, 2)
    image_data = torch.from_numpy(image_data).type(torch.float)
    image_size = torch.from_numpy(image_size).type(torch.float)
    inputs = [image_data, image_size]

    # Load onnx model
    load_path = "third_party/confidential_customer_models/model_2/onnx/saved/yolo_v3/tiny-yolov3-11.onnx"
    model_name = f"yolov3_tiny_onnx"
    onnx_model = onnx.load(load_path)
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
@pytest.mark.nightly
def test_yolov3_onnx(forge_property_recorder):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="yolov_3",
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Image preprocessing
    pil_img = Image.open("third_party/confidential_customer_models/model_2/onnx/saved/yolo_v3/carvana.jpg")
    # input
    image_data = preprocess(pil_img)
    image_size = np.array([pil_img.size[1], pil_img.size[0]], dtype=np.int32).reshape(1, 2)
    image_data = torch.from_numpy(image_data).type(torch.float)
    image_size = torch.from_numpy(image_size).type(torch.float)
    inputs = [image_data, image_size]

    # Load onnx model
    load_path = "third_party/confidential_customer_models/model_2/onnx/saved/yolo_v3/yolov3-10.onnx"
    model_name = f"yolov3_onnx"
    onnx_model = onnx.load(load_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
