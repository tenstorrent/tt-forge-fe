# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import urllib
import pytest
from PIL import Image
import requests
from loguru import logger

import torch

from transformers import AutoImageProcessor

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from test.utils import download_model
from test.models.utils import build_module_name, Framework, Source
from forge.verify.verify import verify


def generate_model_mobilenetV3_imgcls_torchhub_pytorch(variant):
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", variant, pretrained=True)

    # Run inference on Tenstorrent device
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # TODO : Choose image preprocessor from torchvision, to make a compatible postprocessing of the predicted class
    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    image_tensor = preprocessor(images=image, return_tensors="pt").pixel_values

    return model, [image_tensor], {}


variants = ["mobilenet_v3_large", "mobilenet_v3_small"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mobilenetv3_basic(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="mobilenetv3", variant=variant)

    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_mobilenetV3_imgcls_torchhub_pytorch(
        variant,
    )

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


def generate_model_mobilenetV3_imgcls_timm_pytorch(variant):
    # Both options are good
    # model = timm.create_model('mobilenetv3_small_100', pretrained=True)
    if variant == "mobilenetv3_small_100":
        model = download_model(timm.create_model, f"hf_hub:timm/mobilenetv3_small_100.lamb_in1k", pretrained=True)
    else:
        model = download_model(timm.create_model, f"hf_hub:timm/mobilenetv3_large_100.ra_in1k", pretrained=True)

    # Image load and pre-processing into pixel_values
    try:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        url, filename = (
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg",
        )
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")
        image_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image_tensor = torch.rand(1, 3, 224, 224)

    return model, [image_tensor], {}


variants = ["mobilenetv3_large_100", "mobilenetv3_small_100"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mobilenetv3_timm(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="mobilnetv3", source=Source.TIMM, variant=variant
    )

    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_mobilenetV3_imgcls_timm_pytorch(
        variant,
    )

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)
