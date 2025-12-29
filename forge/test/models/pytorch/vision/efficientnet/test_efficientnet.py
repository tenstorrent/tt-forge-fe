# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.efficientnet.pytorch import ModelLoader, ModelVariant

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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

## https://huggingface.co/docs/timm/models/efficientnet

# TIMM variants using loader enums
variants = [
    pytest.param(ModelVariant.TIMM_EFFICIENTNET_B0, id="efficientnet_b0"),
    pytest.param(ModelVariant.TIMM_EFFICIENTNET_B4, id="efficientnet_b4"),
    pytest.param(ModelVariant.HF_TIMM_EFFICIENTNET_B0_RA_IN1K, id="hf_hub_timm_efficientnet_b0_ra_in1k"),
    pytest.param(ModelVariant.HF_TIMM_EFFICIENTNET_B4_RA2_IN1K, id="hf_hub_timm_efficientnet_b4_ra2_in1k"),
    pytest.param(ModelVariant.HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K, id="hf_hub_timm_efficientnet_b5_in12k_ft_in1k"),
    pytest.param(ModelVariant.HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K, id="hf_hub_timm_tf_efficientnet_b0_aa_in1k"),
    pytest.param(ModelVariant.HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K, id="hf_hub_timm_efficientnetv2_rw_s_ra2_in1k"),
    pytest.param(ModelVariant.HF_TIMM_TF_EFFICIENTNETV2_S_IN21K, id="hf_hub_timm_tf_efficientnetv2_s_in21k"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_timm(variant):
    if variant == ModelVariant.TIMM_EFFICIENTNET_B0:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.EFFICIENTNET,
        variant=variant.value,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLASSIFICATION,
        group=group,
        priority=priority,
    )

    # Load model and inputs using ModelLoader
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

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
    if variant in [ModelVariant.HF_TIMM_EFFICIENTNET_B0_RA_IN1K, ModelVariant.TIMM_EFFICIENTNET_B0]:
        pcc = 0.98

    # Model Verification
    _, co_out = verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    # Run model on sample data and print results
    loader.print_cls_results(co_out)


variants = [
    ModelVariant.B0,
    ModelVariant.B1,
    ModelVariant.B2,
    ModelVariant.B3,
    ModelVariant.B4,
    ModelVariant.B5,
    ModelVariant.B6,
    ModelVariant.B7,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.EFFICIENTNET,
        variant=variant.value,
        source=Source.TORCHVISION,
        task=Task.CV_IMAGE_CLASSIFICATION,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.print_cls_results(co_out)
