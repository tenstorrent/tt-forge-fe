# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.beit.pytorch import ModelLoader as BeitLoader
from third_party.tt_forge_models.beit.pytorch import ModelVariant as BeitVariant

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

BEIT_VARIANTS = [
    pytest.param(BeitVariant.BASE, marks=[pytest.mark.xfail]),
    pytest.param(BeitVariant.LARGE, marks=[pytest.mark.xfail]),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", BEIT_VARIANTS)
def test_beit_image_classification_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BEIT,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.GENERALITY,
        priority=ModelPriority.P2,
    )

    # Load model and input
    loader = BeitLoader(variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    model.config.return_dict = False
    input_dict = loader.load_inputs(dtype_override=torch.bfloat16)

    # prepare inputs
    inputs = [input_dict["pixel_values"]]
    model.eval()

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(model, inputs, module_name, compiler_cfg=compiler_cfg)

    # Model Verification
    verify(inputs, model, compiled_model)
