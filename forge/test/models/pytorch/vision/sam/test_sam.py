# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.sam.model_utils.model import (
    SamWrapper,
    get_model_inputs,
)


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("facebook/sam-vit-huge", marks=pytest.mark.skip(reason="Skipping due to CI/CD Limitations")),
        pytest.param("facebook/sam-vit-large", marks=pytest.mark.skip(reason="Skipping due to CI/CD Limitations")),
        pytest.param("facebook/sam-vit-base", marks=pytest.mark.xfail()),
    ],
)
@pytest.mark.nightly
def test_sam(variant):
    group = ModelGroup.RED if variant == "facebook/sam-vit-base" else ModelGroup.GENERALITY

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="sam",
        variant=variant,
        task=Task.IMAGE_SEGMENTATION,
        source=Source.GITHUB,
        group=group,
    )

    # Load  model and input
    framework_model, sample_inputs = get_model_inputs(variant)
    sample_inputs = [sample_inputs[0].to(torch.bfloat16), sample_inputs[1].to(torch.bfloat16)]

    # Forge compile framework model
    framework_model = SamWrapper(framework_model).to(torch.bfloat16)

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
