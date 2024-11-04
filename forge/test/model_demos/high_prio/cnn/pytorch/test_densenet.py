# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
import urllib.request
from loguru import logger

import skimage
import torch
import torchvision
import torchvision.transforms

import torchxrayvision as xrv

import torch

from PIL import Image
import urllib
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize, CenterCrop
import os

############
def get_input_img():
    try:
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")

        transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                PILToTensor(),
                ConvertImageDtype(torch.float32),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Preprocessing
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)
    print(img_tensor.shape)
    return img_tensor


#############


def get_input_img_hf_xray():
    try:
        img_url = "https://huggingface.co/spaces/torchxrayvision/torchxrayvision-classifier/resolve/main/16747_3_1.jpg"
        img_path = "xray.jpg"
        urllib.request.urlretrieve(img_url, img_path)
        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255)
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")
        # Add color channel
        img = img[None, :, :]
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        img = transform(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 1, 224, 224)

    return img_tensor


variants = ["densenet121", "densenet121_hf_xray"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_densenet_121_pytorch(variant, test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.GENERATE_INITIAL_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    if variant == "densenet121":
        model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet121", pretrained=True)
        img_tensor = get_input_img()
    else:
        model_name = "densenet121-res224-all"
        model = download_model(xrv.models.get_model, model_name)
        img_tensor = get_input_img_hf_xray()

    # STEP 3: Run inference on Tenstorrent device
    model(img_tensor)
    inputs = [img_tensor]
    variant_name = variant.replace("-", "_")
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=f"pt_{variant_name}")


def test_densenet_161_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.GENERATE_INITIAL_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet161", pretrained=True)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()
    model(img_tensor)
    inputs = [img_tensor]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_densenet_161")


def test_densenet_169_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.GENERATE_INITIAL_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet169", pretrained=True)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()
    model(img_tensor)
    inputs = [img_tensor]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_densenet_169")


def test_densenet_201_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.GENERATE_INITIAL_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet201", pretrained=True)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()
    model(img_tensor)
    inputs = [img_tensor]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_densenet_201")
