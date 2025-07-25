# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.deit.pytorch import ModelLoader, ModelVariant

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
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

variants = [ModelVariant.BASE, ModelVariant.BASE_DISTILLED, ModelVariant.SMALL, ModelVariant.TINY]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_deit_imgcls_hf_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DEIT,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    inputs_dict = loader.load_inputs(dtype_override=torch.bfloat16)

    # Extract pixel_values from inputs dict
    inputs = [inputs_dict["pixel_values"]]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    pcc = 0.99
    if variant == ModelVariant.BASE:
        pcc = 0.96

    # Model Verification and inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post processing
    loader.post_processing(co_out, framework_model)
