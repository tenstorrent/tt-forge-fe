# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.pytorch.vision.rmbg.model_utils.utils import load_input, load_model


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["briaai/RMBG-2.0"])
def test_rmbg(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="rmbg_2_0",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_SEGMENTATION,
    )

    # Load model and input
    framework_model = load_model(variant).to(torch.bfloat16)
    inputs = load_input()

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
