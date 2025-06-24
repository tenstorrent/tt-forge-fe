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

from test.models.pytorch.vision.ghostnet.model_utils.utils import (
    load_ghostnet_model,
    post_processing,
)

params = [
    pytest.param("ghostnet_100", marks=[pytest.mark.push]),
    pytest.param("ghostnet_100.in1k", marks=[pytest.mark.push]),
    pytest.param(
        "ghostnetv2_100.in1k",
        marks=[pytest.mark.xfail],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", params)
def test_ghostnet_timm(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GHOSTNET,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and prepare input data
    framework_model, inputs = load_ghostnet_model(variant)
    framework_model.to(torch.bfloat16)
    inputs = [inputs[0].to(torch.bfloat16)]

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

    # Post processing
    post_processing(co_out)
