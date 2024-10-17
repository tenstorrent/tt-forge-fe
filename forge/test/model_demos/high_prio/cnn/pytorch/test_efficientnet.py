# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.utils import download_model
import timm
import pytest
import urllib
import torch
from PIL import Image
import torchvision.models as models
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from loguru import logger

import forge

## https://huggingface.co/docs/timm/models/efficientnet

variants = [
    "efficientnet_b0",
    "efficientnet_b4",
    # "hf_hub:timm/efficientnet_b0.ra_in1k",
    # "hf_hub:timm/efficientnet_b4.ra2_in1k",
    # "hf_hub:timm/efficientnet_b5.in12k_ft_in1k",
    # "hf_hub:timm/tf_efficientnet_b0.aa_in1k",
    # "hf_hub:timm/efficientnetv2_rw_s.ra2_in1k",
    # "hf_hub:timm/tf_efficientnetv2_s.in21k",
]


@pytest.mark.parametrize("variant", variants)
def test_efficientnet_timm(variant, test_device):

    # Configuration
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Load model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # Load and pre-process image
    try:
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        config = resolve_data_config({}, model=framework_model)
        transform = create_transform(**config)
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    compiled_model = forge.compile(framework_model, sample_inputs=[img_tensor])


variants = [
    models.efficientnet_b0,
    # models.efficientnet_b1,
    # models.efficientnet_b2,
    # models.efficientnet_b3,
    models.efficientnet_b4,
    # models.efficientnet_b5,
    # models.efficientnet_b6,
    # models.efficientnet_b7,
]


@pytest.mark.skip(reason="invalid hash value")
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_torchvision(variant, test_device):
    # Configuration
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Load model
    framework_model = download_model(variant, pretrained=True)
    framework_model.eval()
    # Load and pre-process image
    try:
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        config = resolve_data_config({}, model=framework_model)
        transform = create_transform(**config)
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    compiled_model = forge.compile(framework_model, sample_inputs=[img_tensor])
