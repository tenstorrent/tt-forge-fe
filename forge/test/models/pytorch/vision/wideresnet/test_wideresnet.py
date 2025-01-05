# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
from test.utils import download_model
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
import urllib
import os
from test.models.utils import build_module_name


def generate_model_wideresnet_imgcls_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    img_tensor = input_tensor.unsqueeze(0)

    return framework_model, [img_tensor]


variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_wideresnet_pytorch(variant):
    (model, inputs,) = generate_model_wideresnet_imgcls_pytorch(
        variant,
    )

    module_name = build_module_name(framework="pt", model="wideresnet_pytorch", variant=variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


def generate_model_wideresnet_imgcls_timm(variant):
    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return framework_model, [img_tensor]


variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_wideresnet_timm(variant):
    (model, inputs,) = generate_model_wideresnet_imgcls_timm(
        variant,
    )

    module_name = build_module_name(framework="pt", model="wideresnet", source="timm", variant=variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
