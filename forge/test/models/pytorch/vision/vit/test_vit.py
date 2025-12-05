# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.vit.pytorch import ModelLoader, ModelVariant

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
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

variants = [
    ModelVariant.BASE,
    ModelVariant.LARGE,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vit_classify_224_hf_pytorch(variant):

    # Record Forge Property
    if variant == ModelVariant.BASE:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VIT,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
        group=group,
        priority=priority,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model.config.return_dict = False
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [inputs]

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.post_processing(co_out)


variants = [
    ModelVariant.VIT_B_16,
    ModelVariant.VIT_B_32,
    ModelVariant.VIT_L_16,
    ModelVariant.VIT_L_32,
    ModelVariant.VIT_H_14,
]


@pytest.mark.nightly
@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2998")
@pytest.mark.parametrize("variant", variants)
def test_vit_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VIT,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [inputs]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    pcc = 0.99

    if variant in [ModelVariant.VIT_B_32, ModelVariant.VIT_L_32]:
        pcc = 0.98
    elif variant == ModelVariant.VIT_H_14:
        pcc = 0.93

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Run model on sample data and print results
    loader.post_processing(co_out)
