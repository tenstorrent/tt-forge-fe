# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## Inception V4

# STEP 0: import Forge library
import forge
import os
import urllib
from loguru import logger
from test.utils import download_model
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from PIL import Image
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

torch.multiprocessing.set_sharing_strategy("file_system")
import forge


def generate_model_inceptionV4_imgcls_osmr_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Load model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)

    # Load and pre-process image
    img_tensor = get_image()

    return framework_model, [img_tensor]


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
        img_tensor = torch.rand(1, 3, 299, 299)
    return model, img_tensor


def get_image():
    try:
        if not os.path.exists("dog.jpg"):
            torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(input_image)
        img_tensor = img_tensor.unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 299, 299)
    return img_tensor


def test_inception_v4_osmr_pytorch(test_device):
    model, inputs = generate_model_inceptionV4_imgcls_osmr_pytorch("inceptionv4")
    compiled_model = forge.compile(model, sample_inputs=inputs)


def generate_model_inceptionV4_imgcls_timm_pytorch(test_device, variant):
    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Load model & Preprocess image
    framework_model, img_tensor = download_model(preprocess_timm_model, variant)
    return framework_model, [img_tensor]


def test_inception_v4_timm_pytorch(test_device):
    model, inputs = generate_model_inceptionV4_imgcls_timm_pytorch("inception_v4")

    compiled_model = forge.compile(model, sample_inputs=inputs)
