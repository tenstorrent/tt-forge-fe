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

# TODO: These are old forge, we should update them to the currently version.
# import forge
# from forge.verify.backend import verify_module
# from forge import DepricatedVerifyConfig, PyTorchModule
# from forge._C.backend_api import BackendType, BackendDevice
# from forge.verify.config import TestKind


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
@pytest.mark.skip(reason="Requires restructuring")
@pytest.mark.nightly
def test_retinanet_r101_640x480_onnx(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b

    # STEP 2: Create Forge module from PyTorch model
    load_path = "third_party/confidential_customer_models/model_2/onnx/retinanet/retinanet-9.onnx"
    model = onnx.load(load_path)
    tt_model = forge.OnnxModule("onnx_retinanet", model)

    # Image preprocessing
    img_tensor = img_preprocess()

    # STEP 3: Run inference on Tenstorrent device
    pcc = 0.97 if test_device.arch == BackendDevice.Grayskull and test_device.devtype == BackendType.Silicon else 0.99
    verify_module(
        tt_model,
        input_shapes=([img_tensor.shape]),
        inputs=([img_tensor]),
        verify_cfg=DepricatedVerifyConfig(
            test_kind=TestKind.INFERENCE,
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            pcc=pcc,
        ),
    )


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
@pytest.mark.skip(reason="Requires restructuring")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_retinanet_onnx(variant, test_device):

    # Set Forge configuration parameters
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b

    # Prepare model
    load_path = f"third_party/confidential_customer_models/generated/files/{variant}.onnx"
    model_name = f"onnx_{variant}"
    model = onnx.load(load_path)
    tt_model = forge.OnnxModule(model_name, model)

    # Prepare input
    input_batch = img_preprocessing()

    # Inference
    verify_module(
        tt_model,
        input_shapes=([input_batch.shape]),
        inputs=([input_batch]),
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
