# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.xception.model_utils.utils import post_processing
from test.utils import download_model


def generate_model_xception_imgcls_timm(variant):
    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)

    img = Image.open(requests.get("https://github.com/pytorch/hub/raw/master/images/dog.jpg", stream=True).raw).convert(
        "RGB"
    )
    img_tensor = transform(img).unsqueeze(0)

    return framework_model.to(torch.bfloat16), [img_tensor.to(torch.bfloat16)]


params = [
    pytest.param(
        "xception",
        marks=[pytest.mark.xfail],
    ),
    pytest.param("xception41"),
    pytest.param("xception65"),
    pytest.param("xception71"),
    pytest.param("xception71.tf_in1k", marks=[pytest.mark.push]),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", params)
def test_xception_timm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.XCEPTION,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    (framework_model, inputs) = generate_model_xception_imgcls_timm(variant)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Post Processing
    if variant == "xception71.tf_in1k":
        post_processing(co_out)
