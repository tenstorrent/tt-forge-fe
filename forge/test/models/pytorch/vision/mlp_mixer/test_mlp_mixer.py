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
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.utils import download_model

varaints = [
    pytest.param(
        "mixer_b16_224",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "mixer_b16_224_in21k",
        marks=[pytest.mark.xfail],
    ),
    pytest.param("mixer_b16_224_miil"),
    pytest.param(
        "mixer_b16_224_miil_in21k",
        marks=[pytest.mark.xfail],
    ),
    pytest.param("mixer_b32_224"),
    pytest.param(
        "mixer_l16_224",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "mixer_l16_224_in21k",
        marks=[pytest.mark.xfail],
    ),
    pytest.param("mixer_l32_224"),
    pytest.param("mixer_s16_224"),
    pytest.param("mixer_s32_224"),
    pytest.param(
        "mixer_b16_224.goog_in21k",
        marks=[pytest.mark.xfail],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", varaints)
def test_mlp_mixer_timm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="mlp_mixer",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    load_pretrained_weights = True
    if variant in ["mixer_s32_224", "mixer_s16_224", "mixer_b32_224", "mixer_l32_224"]:
        load_pretrained_weights = False

    framework_model = download_model(timm.create_model, variant, pretrained=load_pretrained_weights).to(torch.bfloat16)
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

    inputs = [pixel_values.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


@pytest.mark.nightly
@pytest.mark.xfail
def test_mlp_mixer_pytorch():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="mlp_mixer",
        source=Source.GITHUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and input
    framework_model = MLPMixer(
        image_size=256,
        channels=3,
        patch_size=16,
        dim=512,
        depth=12,
        num_classes=1000,
    ).to(torch.bfloat16)
    framework_model.eval()

    inputs = [torch.randn(1, 3, 256, 256).to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
