# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# STEP 0: import Forge library
import pytest
import os

import onnx
import torch
import tensorflow as tf

from PIL import Image
import numpy as np

import requests
from torchvision import transforms

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task


## https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/retinanet

########
def img_preprocess(scal_val=1):
    pil_img = Image.open("forge/test/models/files/samples/images/carvana.jpg")
    scale = scal_val
    w, h = pil_img.size
    print("----", w, h)
    newW, newH = int(scale * w), int(scale * h)
    newW, newH = 640, 480
    assert newW > 0 and newH > 0, "Scale is too small, resized images would have no pixel"
    pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(pil_img, dtype=np.float32)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))
    if (img > 1).any():
        img = img / 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    return img


#########


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.nightly
def test_retinanet_r101_640x480_onnx(forge_property_recorder):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="retinanet",
        source=Source.HUGGINGFACE,
        task=Task.OBJECT_DETECTION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Image preprocessing
    img_tensor = img_preprocess()
    inputs = [img_tensor]

    # Load onnx model
    load_path = "third_party/confidential_customer_models/model_2/onnx/retinanet/retinanet-9.onnx"
    model_name = f"retinanet_r101_640x480_onnx"
    onnx_model = onnx.load(load_path)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def img_preprocessing():

    url = "https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg"
    pil_img = Image.open(requests.get(url, stream=True).raw)
    new_size = (640, 480)
    pil_img = pil_img.resize(new_size, resample=Image.BICUBIC)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = preprocess(pil_img)
    img = img.unsqueeze(0)
    return img


variants = [
    "retinanet_rn18fpn",
    "retinanet_rn34fpn",
    "retinanet_rn50fpn",
    "retinanet_rn152fpn",
]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_retinanet_onnx(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="retinanet",
        source=Source.HUGGINGFACE,
        variant=variant,
        task=Task.OBJECT_DETECTION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Prepare input
    input_batch = img_preprocessing()
    inputs = [input_batch]

    # Load onnx model
    load_path = f"third_party/confidential_customer_models/generated/files/{variant}.onnx"
    model_name = f"retinanet_{variant}_onnx"
    onnx_model = onnx.load(load_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
