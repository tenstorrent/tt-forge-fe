# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import torch
import forge
import os


def generate_model_yoloV5I320_imgcls_torchhub_pytorch(test_device, variant, size):
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    name = "yolov5" + size

    model = download_model(torch.hub.load, variant, name, pretrained=True)

    input_shape = (1, 3, 320, 320)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = ["n", "s", "m", "l", "x"]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_320x320(test_device, size):
    model, inputs, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        test_device,
        "ultralytics/yolov5",
        size=size,
    )
    ouputs = model(inputs[0])
    name = "yolov5" + size
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_" + name + "_320x320")


def generate_model_yoloV5I640_imgcls_torchhub_pytorch(test_device, variant, size):
    # env vars needed to support 640x640 yolov5 working
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)

    input_shape = (1, 3, 640, 640)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = ["n", "s", "m", "l", "x"]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_640x640(test_device, size):

    model, inputs, _ = generate_model_yoloV5I640_imgcls_torchhub_pytorch(
        test_device,
        "ultralytics/yolov5",
        size=size,
    )
    ouputs = model(inputs[0])
    name = "yolov5" + size
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_" + name + "_640x640")


def generate_model_yoloV5I480_imgcls_torchhub_pytorch(test_device, variant, size):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    name = "yolov5" + size
    model = download_model(torch.hub.load, variant, name, pretrained=True)
    input_shape = (1, 3, 480, 480)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


@pytest.mark.nightly
@pytest.mark.parametrize("size", size, ids=["yolov5" + s for s in size])
def test_yolov5_480x480(test_device, size):

    model, inputs, _ = generate_model_yoloV5I480_imgcls_torchhub_pytorch(
        test_device,
        "ultralytics/yolov5",
        size=size,
    )
    ouputs = model(inputs[0])
    name = "yolov5" + size
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_" + name + "_480x480")


@pytest.mark.nightly
def test_yolov5_1280x1280(test_device):

    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    model = download_model(torch.hub.load, "ultralytics/yolov5", "yolov5s", pretrained=True)

    input_shape = (1, 3, 1280, 1280)
    input_tensor = torch.rand(input_shape)
    inputs = [input_tensor]
    ouputs = model(inputs[0])
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_yolov5s_1280x1280")
