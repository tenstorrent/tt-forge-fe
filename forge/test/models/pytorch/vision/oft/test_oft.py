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
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from third_party.tt_forge_models.oft import ModelLoader  # isort:skip


@pytest.mark.nightly
@pytest.mark.xfail
def test_oft():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.OFT,
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Load model and input
    framework_model = ModelLoader.load_model()
    framework_model.to(torch.bfloat16)
    input_sample = ModelLoader.load_inputs()
    input_sample_1, input_sample_2, input_sample_3 = input_sample
    inputs = [input_sample_1.to(torch.bfloat16), input_sample_2.to(torch.bfloat16), input_sample_3.to(torch.bfloat16)]

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(input_sample, framework_model, compiled_model)
