# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import timm
import torch
from loguru import logger
from mlp_mixer_pytorch import MLPMixer
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

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
@pytest.mark.parametrize("variant", varaints, ids=varaints)
def test_mlp_mixer_timm_pytorch(record_forge_property, variant):
    if variant != "mixer_b16_224":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mlp_mixer",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    framework_model = download_model(timm.create_model, variant, pretrained=True)
    config = resolve_data_config({}, model=framework_model)
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

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
def test_mlp_mixer_pytorch(record_forge_property):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mlp_mixer",
        source=Source.GITHUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    # Load model and input
    framework_model = MLPMixer(
        image_size=256,
        channels=3,
        patch_size=16,
        dim=512,
        depth=12,
        num_classes=1000,
    )
    framework_model.eval()

    inputs = [torch.randn(1, 3, 256, 256)]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
