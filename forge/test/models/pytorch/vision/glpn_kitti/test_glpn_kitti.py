# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.glpn_kitti.pytorch import ModelLoader

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


@pytest.mark.nightly
@pytest.mark.xfail
def test_glpn_kitti():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GLPNKITTI,
        variant="default",
        source=Source.HUGGINGFACE,
        task=Task.CV_DEPTH_ESTIMATION,
    )

    # Load model and input
    loader = ModelLoader()
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_dict = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_dict["pixel_values"]]

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
