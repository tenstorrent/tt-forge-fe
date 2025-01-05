# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from loguru import logger
from PIL import Image
import urllib

import torch
from torchvision import transforms

from pytorchcv.model_provider import get_model as ptcv_get_model

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from test.models.utils import build_module_name, Framework, Task, Source

from test.utils import download_model

torch.multiprocessing.set_sharing_strategy("file_system")


def generate_model_hrnet_imgcls_osmr_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    """
    models = [
        hrnet_w18_small_v1,
        hrnet_w18_small_v2,
        hrnetv2_w18,
        hrnetv2_w30,
        hrnetv2_w32,
        hrnetv2_w40,
        hrnetv2_w44,
        hrnetv2_w48,
        hrnetv2_w64,
    ]
    """
    model = download_model(ptcv_get_model, variant, pretrained=True)
    model.eval()

    # Model load
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
    print(input_batch.shape)

    return model, [input_batch], {}


variants = [
    "hrnet_w18_small_v1",
    "hrnet_w18_small_v2",
    "hrnetv2_w18",
    "hrnetv2_w30",
    "hrnetv2_w32",
    "hrnetv2_w40",
    "hrnetv2_w44",
    "hrnetv2_w48",
    "hrnetv2_w64",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_hrnet_osmr_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="hrnet", variant=variant, source=Source.OSMR)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_hrnet_imgcls_osmr_pytorch(
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)


def generate_model_hrnet_imgcls_timm_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    """
    default_cfgs = {
    'hrnet_w18_small'
    'hrnet_w18_small_v2'
    'hrnet_w18'
    'hrnet_w30'
    'hrnet_w32'
    'hrnet_w40'
    'hrnet_w44'
    'hrnet_w48'
    'hrnet_w64'
    }
    """
    model = download_model(timm.create_model, variant, pretrained=True)
    model.eval()

    ## Preprocessing
    try:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_tensor = torch.rand(1, 3, 224, 224)
    print(input_tensor.shape)

    return model, [input_tensor], {}


variants = [
    "hrnet_w18_small",
    "hrnet_w18_small_v2",
    "hrnet_w18",
    "hrnet_w30",
    "hrnet_w32",
    "hrnet_w40",
    "hrnet_w44",
    "hrnet_w48",
    "hrnet_w64",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_hrnet_timm_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="hrnet", variant=variant, source=Source.TIMM)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_hrnet_imgcls_timm_pytorch(
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)
