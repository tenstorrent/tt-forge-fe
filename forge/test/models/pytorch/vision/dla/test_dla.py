# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

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

from test.models.pytorch.vision.dla.model_utils.utils import load_dla_model
from test.models.pytorch.vision.vision_utils.utils import load_timm_model_and_input

variants = [
    "dla34",
    "dla46_c",
    "dla46x_c",
    "dla60",
    "dla60x",
    "dla60x_c",
    "dla102",
    "dla102x",
    "dla102x2",
    "dla169",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_dla_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DLA,
        variant=variant,
        task=Task.VISUAL_BACKBONE,
        source=Source.TORCHVISION,
    )

    # Load the model and prepare input data
    framework_model, inputs = load_dla_model(variant)
    framework_model.to(torch.bfloat16)
    inputs = [inputs[0].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = ["dla34.in1k"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_dla_timm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DLA,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and inputs
    framework_model, inputs = load_timm_model_and_input(variant)
    framework_model.to(torch.bfloat16)
    inputs = [inputs.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
