# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.sam.pytorch import ModelLoader, ModelVariant

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.sam.model_utils.model import SamWrapper

variants = [
    pytest.param(
        ModelVariant.HUGE,
        marks=[pytest.mark.skip_model_analysis],
    ),
    pytest.param(
        ModelVariant.LARGE,
        marks=[pytest.mark.skip_model_analysis],
    ),
    pytest.param(ModelVariant.BASE, marks=pytest.mark.xfail()),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_sam(variant):

    if variant == ModelVariant.BASE:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SAM,
        variant=variant,
        task=Task.IMAGE_SEGMENTATION,
        source=Source.GITHUB,
        group=group,
        priority=priority,
    )
    if variant != ModelVariant.BASE:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model = SamWrapper(framework_model)
    pixel_values, input_points = loader.load_inputs(dtype_override=torch.bfloat16)
    sample_inputs = [pixel_values, input_points]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=sample_inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(sample_inputs, framework_model, compiled_model)
