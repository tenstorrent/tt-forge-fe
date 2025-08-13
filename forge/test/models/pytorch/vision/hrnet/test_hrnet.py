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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify
from third_party.tt_forge_models.hrnet.pytorch import ModelLoader, ModelVariant

# OSMR (pytorchcv) variants using loader enums
variants = [
    pytest.param(ModelVariant.HRNET_W18_SMALL_V1_OSMR, marks=pytest.mark.push),
    pytest.param(ModelVariant.HRNET_W18_SMALL_V2_OSMR),
    pytest.param(ModelVariant.HRNETV2_W18_OSMR),
    pytest.param(ModelVariant.HRNETV2_W30_OSMR),
    pytest.param(ModelVariant.HRNETV2_W32_OSMR),
    pytest.param(ModelVariant.HRNETV2_W40_OSMR),
    pytest.param(ModelVariant.HRNETV2_W44_OSMR),
    pytest.param(ModelVariant.HRNETV2_W48_OSMR),
    pytest.param(ModelVariant.HRNETV2_W64_OSMR),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_hrnet_osmr_pytorch(variant):

    pcc = 0.99
    if variant == ModelVariant.HRNETV2_W44_OSMR:
        pcc = 0.97
    if variant in [ModelVariant.HRNETV2_W64_OSMR, ModelVariant.HRNETV2_W40_OSMR, ModelVariant.HRNETV2_W30_OSMR, ModelVariant.HRNETV2_W32_OSMR]:
        pcc = 0.95

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.HRNET,
        variant=variant.value,
        source=Source.OSMR,
        task=Task.IMAGE_CLASSIFICATION,
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

    # Model Verification
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Run model on sample data and print results
    loader.print_cls_results(co_out)


variants = [
    ModelVariant.HRNET_W18_SMALL,
    ModelVariant.HRNET_W18_SMALL_V2,
    ModelVariant.HRNET_W18,
    ModelVariant.HRNET_W30,
    ModelVariant.HRNET_W32,
    ModelVariant.HRNET_W40,
    ModelVariant.HRNET_W44,
    ModelVariant.HRNET_W48,
    ModelVariant.HRNET_W64,
    ModelVariant.HRNET_W18_MS_AUG_IN1K,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_hrnet_timm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.HRNET,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    if variant in [
        ModelVariant.HRNET_W32,
        ModelVariant.HRNET_W40,
        ModelVariant.HRNET_W44,
        ModelVariant.HRNET_W48,
        ModelVariant.HRNET_W64,
        ModelVariant.HRNET_W18_MS_AUG_IN1K,
    ]:
        pytest.xfail(reason="Requires multi-chip support")

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

    # Run model on sample data and print results
    loader.print_cls_results(co_out)
