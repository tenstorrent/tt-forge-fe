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
from torch.hub import load_state_dict_from_url
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    efficientnet_b0,
    efficientnet_b4,
)
from torchvision.models._api import WeightsEnum

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.utils import download_model

## https://huggingface.co/docs/timm/models/efficientnet

variants = [
    pytest.param(
        "efficientnet_b0",
        id="efficientnet_b0",
        marks=[pytest.mark.push, pytest.mark.models],
    ),
    pytest.param(
        "efficientnet_b4",
        id="efficientnet_b4",
    ),
    # pytest.param("hf_hub:timm/efficientnet_b0.ra_in1k", id="hf_hub_timm_efficientnet_b0_ra_in1k"),
    # pytest.param("hf_hub:timm/efficientnet_b4.ra2_in1k", id="hf_hub_timm_efficientnet_b4_ra2_in1k"),
    # pytest.param("hf_hub:timm/efficientnet_b5.in12k_ft_in1k", id="hf_hub_timm_efficientnet_b5_in12k_ft_in1k"),
    # pytest.param("hf_hub:timm/tf_efficientnet_b0.aa_in1k", id="hf_hub_timm_tf_efficientnet_b0_aa_in1k"),
    # pytest.param("hf_hub:timm/efficientnetv2_rw_s.ra2_in1k", id="hf_hub_timm_efficientnetv2_rw_s_ra2_in1k"),
    # pytest.param("hf_hub:timm/tf_efficientnetv2_s.in21k", id="hf_hub_timm_tf_efficientnetv2_s_in21k"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_timm(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="efficientnet",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    if variant in ["efficientnet_b0"]:
        forge_property_recorder.record_group("red")
        forge_property_recorder.record_priority("P1")
    else:
        forge_property_recorder.record_group("generality")

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

    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict

variants = [
    "efficientnet_b0",
    # models.efficientnet_b1,
    # models.efficientnet_b2,
    # models.efficientnet_b3,
    "efficientnet_b4",
    # models.efficientnet_b5,
    # models.efficientnet_b6,
    # models.efficientnet_b7,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_torchvision(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="efficientnet",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load model
    if variant == "efficientnet_b0":
        framework_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif variant == "efficientnet_b4":
        framework_model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

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

    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
