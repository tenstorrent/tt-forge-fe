# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.utils import download_model
import timm
import pytest
import urllib
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from loguru import logger
import forge
from torchvision.models import efficientnet_b4, efficientnet_b0, EfficientNet_B4_Weights, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name, Framework, Source
from forge.verify.verify import verify

## https://huggingface.co/docs/timm/models/efficientnet

variants = [
    pytest.param(
        "efficientnet_b0",
        id="efficientnet_b0",
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
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_timm(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="efficientnet", variant=variant, source=Source.TIMM
    )

    record_forge_property("module_name", module_name)

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

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


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
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_torchvision(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="efficientnet", variant=variant, source=Source.TORCHVISION
    )

    record_forge_property("module_name", module_name)

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

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)
