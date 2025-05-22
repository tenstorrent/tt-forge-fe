# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## Inception V4
import pytest
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.inception.model_utils.model_utils import (
    get_image,
    preprocess_timm_model,
)
from test.utils import download_model


def generate_model_inceptionV4_imgcls_osmr_pytorch(variant):
    # Load model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)

    # Load and pre-process image
    img_tensor = get_image()

    return framework_model.to(torch.bfloat16), [img_tensor.to(torch.bfloat16)]


@pytest.mark.nightly
@pytest.mark.xfail
def test_inception_v4_osmr_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="inception", variant="v4", source=Source.OSMR, task=Task.IMAGE_CLASSIFICATION
    )

    framework_model, inputs = generate_model_inceptionV4_imgcls_osmr_pytorch("inceptionv4")

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


def generate_model_inceptionV4_imgcls_timm_pytorch(variant):
    # Load model & Preprocess image
    framework_model, img_tensor = download_model(preprocess_timm_model, variant)
    return framework_model.to(torch.bfloat16), [img_tensor.to(torch.bfloat16)]


variants = [
    pytest.param(
        "inception_v4",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "inception_v4.tf_in1k",
        marks=[pytest.mark.xfail],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_inception_v4_timm_pytorch(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="inception",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    framework_model, inputs = generate_model_inceptionV4_imgcls_timm_pytorch(variant)

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
