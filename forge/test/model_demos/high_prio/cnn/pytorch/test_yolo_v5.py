# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import torch
import forge


def generate_model_yoloV5I320_imgcls_torchhub_pytorch(test_device, variant, size):
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    name = "yolov5" + size

    model = download_model(torch.hub.load, variant, name, pretrained=True)

    input_shape = (1, 3, 320, 320)

    return model, [input_shape], {}


size = ["n", "s", "m", "l", "x"]


@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_320x320(test_device, size):
    model, inputs, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        test_device,
        "ultralytics/yolov5",
        size=size,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]])


def generate_model_yoloV5I640_imgcls_torchhub_pytorch(test_device, variant, size):
    # env vars needed to support 640x640 yolov5 working
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)

    input_shape = (1, 3, 640, 640)

    return model, [input_shape], {}


size = ["n", "s", "m", "l", "x"]


@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_640x640(test_device, size):

    model, inputs, _ = generate_model_yoloV5I640_imgcls_torchhub_pytorch(
        test_device,
        "ultralytics/yolov5",
        size=size,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]])


def generate_model_yoloV5I480_imgcls_torchhub_pytorch(test_device, variant, size):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)
    input_shape = (1, 3, 480, 480)

    return model, [input_shape], {}


@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_480x480(test_device, size):

    model, inputs, _ = generate_model_yoloV5I480_imgcls_torchhub_pytorch(
        test_device,
        "ultralytics/yolov5",
        size=size,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]])


@pytest.mark.skip(reason="Not supported")
def test_yolov5_1280x1280(test_device):
    # env vars needed to support 640x640 yolov5 working
    compiler_cfg.paddings = {
        "concatenate_19.dc.concatenate.4": True,
        "concatenate_46.dc.concatenate.4": True,
        "concatenate_139.dc.concatenate.4": True,
        "concatenate_152.dc.concatenate.4": True,
    }

    model = download_model(torch.hub.load, "ultralytics/yolov5", "yolov5s", pretrained=True)
    compiled_model = forge.compile(model, sample_inputs=[input_shape])
