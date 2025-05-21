# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import timm
import torch
from loguru import logger
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.mobilenet.model_utils.utils import (
    load_mobilenet_model,
    post_processing,
)
from test.utils import download_model

variants = [
    pytest.param("mobilenet_v3_large", marks=[pytest.mark.push]),
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

    # Load the model and prepare input data
    framework_model, inputs = load_mobilenet_model(variant)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Post processing
    post_processing(co_out)


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
        img = Image.open(
            requests.get("https://github.com/pytorch/hub/raw/master/images/dog.jpg", stream=True).raw
        ).convert("RGB")
        image_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image_tensor = torch.rand(1, 3, 224, 224)

    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


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

    framework_model, inputs, _ = generate_model_mobilenetV3_imgcls_timm_pytorch(
        variant,
    )

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
