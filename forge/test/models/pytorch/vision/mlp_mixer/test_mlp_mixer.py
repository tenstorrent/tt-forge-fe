# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from PIL import Image
import requests
from loguru import logger

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from test.utils import download_model
from test.models.utils import build_module_name, Framework, Source

import forge


varaints = [
    "mixer_b16_224",
    "mixer_b16_224_in21k",
    "mixer_b16_224_miil",
    "mixer_b16_224_miil_in21k",
    "mixer_b32_224",
    "mixer_l16_224",
    "mixer_l16_224_in21k",
    "mixer_l32_224",
    "mixer_s16_224",
    "mixer_s32_224",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", varaints, ids=varaints)
def test_mlp_mixer_timm_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="mlp_mixer", variant=variant, source=Source.TIMM)

    record_forge_property("module_name", module_name)

    model = download_model(timm.create_model, variant, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    try:
        url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image = torch.rand(1, 3, 256, 256)
    pixel_values = transform(image).unsqueeze(0)
    inputs = [pixel_values]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
