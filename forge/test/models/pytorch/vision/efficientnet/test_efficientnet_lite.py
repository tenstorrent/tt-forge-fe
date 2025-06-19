# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## https://github.com/RangiLyu/EfficientNet-Lite/
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

from test.models.pytorch.vision.vision_utils.utils import load_timm_model_and_input

variants = [
    "tf_efficientnet_lite0.in1k",
    "tf_efficientnet_lite1.in1k",
    "tf_efficientnet_lite2.in1k",
    "tf_efficientnet_lite3.in1k",
    "tf_efficientnet_lite4.in1k",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_lite_timm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.EFFICIENTNETLITE,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and inputs
    framework_model, inputs = load_timm_model_and_input(variant)
    framework_model = framework_model.to(torch.bfloat16)
    inputs = [input.to(torch.bfloat16) for input in inputs]

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
