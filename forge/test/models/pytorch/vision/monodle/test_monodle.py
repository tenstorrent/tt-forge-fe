# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.monodle.pytorch import ModelLoader, ModelVariant

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
def test_monodle_pytorch():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MONODLE,
        variant=ModelVariant.DLA34,
        source=Source.GITHUB,
        task=Task.CV_OBJECT_DETECTION,
    )
    pytest.xfail(reason="Floating point exception(core dumped)")

    # Load model and input
    loader = ModelLoader(ModelVariant.DLA34)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    img_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [img_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
