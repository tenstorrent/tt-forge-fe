# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import urllib

import pytest
import timm
import torch
from loguru import logger
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.mobilenet.utils.utils import (
    load_mobilenet_model,
    post_processing,
)
from test.utils import download_model

variants = [
    pytest.param("mobilenet_v3_large", marks=[pytest.mark.push, pytest.mark.models]),
    pytest.param("mobilenet_v3_small"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mobilenetv3_basic(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilenetv3",
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model and prepare input data
    framework_model, inputs = load_mobilenet_model(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Inference
    output = compiled_model(*inputs)

    # Post processing
    post_processing(output)


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
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mobilenetv3_timm(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilnetv3",
        source=Source.TIMM,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    framework_model, inputs, _ = generate_model_mobilenetV3_imgcls_timm_pytorch(
        variant,
    )

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
