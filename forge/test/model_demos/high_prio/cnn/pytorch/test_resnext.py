# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.utils import download_model
import os
from loguru import logger
import forge
import torch
from PIL import Image
from torchvision import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model


#############
def get_image_tensor():
    # Image processing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)
    return input_batch


def test_resnext_50_torchhub_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "resnext50_32x4d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name="pt_resnext50_torchhub")


def test_resnext_101_torchhub_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "resnext101_32x8d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name="pt_resnext101_torchhub")


def test_resnext_101_32x8d_fb_wsl_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    # STEP 2: Create Forge module from PyTorch model
    # 4 variants
    model = download_model(torch.hub.load, "facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name="pt_resnext101_fb_wsl")


def test_resnext_14_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext14_32x4d", pretrained=True)
    model.eval()
    # tt_model = forge.PyTorchModule("pt_resnext14_osmr", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name="pt_resnext14_osmr")


def test_resnext_26_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext26_32x4d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name="pt_resnext26_osmr")


def test_resnext_50_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext50_32x4d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name="pt_resnext50_osmr")


def test_resnext_101_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext101_64x4d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name="pt_resnext101_osmr")
