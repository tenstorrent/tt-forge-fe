# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model

import forge

from pytorchcv.model_provider import get_model as ptcv_get_model
import os
import torch
from PIL import Image
from torchvision import transforms
from vgg_pytorch import VGG
from loguru import logger
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import urllib
from torchvision import transforms
from test.models.utils import build_module_name


variants = ["vgg11", "vgg13", "vgg16", "vgg19", "bn_vgg19", "bn_vgg19b"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_vgg_osmr_pytorch(variant):
    model = download_model(ptcv_get_model, variant, pretrained=True)
    model.eval()

    # Image preprocessing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
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

    module_name = build_module_name(framework="pt", model="vgg", variant=variant)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_vgg_19_hf_pytorch():
    """
    # https://pypi.org/project/vgg-pytorch/
    # Variants:
    vgg11, vgg11_bn
    vgg13, vgg13_bn
    vgg16, vgg16_bn
    vgg19, vgg19_bn
    """
    model = download_model(VGG.from_pretrained, "vgg19")
    model.eval()

    # Image preprocessing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
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
    module_name = build_module_name(framework="pt", model="vgg", variant="19_hf")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


def preprocess_timm_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    try:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    return model, img_tensor


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_vgg_bn19_timm_pytorch():
    torch.multiprocessing.set_sharing_strategy("file_system")
    model_name = "vgg19_bn"
    model, image_tensor = download_model(preprocess_timm_model, model_name)

    module_name = build_module_name(framework="pt", model="vgg", variant="19_bn", source="timm")
    compiled_model = forge.compile(model, sample_inputs=[image_tensor], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_vgg_bn19_torchhub_pytorch():
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "vgg19_bn", pretrained=True)
    model.eval()

    # Image preprocessing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
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

    module_name = build_module_name(framework="pt", model="vgg", variant="19_bn_torchub")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)
