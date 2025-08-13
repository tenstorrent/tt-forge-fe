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
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify
from third_party.tt_forge_models.dla.pytorch import ModelLoader, ModelVariant

variants = [
    ModelVariant.DLA34,
    ModelVariant.DLA46_C,
    ModelVariant.DLA46X_C,
    ModelVariant.DLA60,
    ModelVariant.DLA60X,
    ModelVariant.DLA60X_C,
    ModelVariant.DLA102,
    ModelVariant.DLA102X,
    ModelVariant.DLA102X2,
    ModelVariant.DLA169,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_dla_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DLA,
        variant=variant,
        task=Task.VISUAL_BACKBONE,
        source=Source.TORCHVISION,
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
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.print_cls_results(co_out)


timm_variants = [
    ModelVariant.DLA34_IN1K,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", timm_variants)
def test_dla_timm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DLA,
        variant=variant.value,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
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
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )
    verify_cfg = VerifyConfig()
    if variant == ModelVariant.DLA34_IN1K:
        verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98))

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Post processing
    loader.print_cls_results(co_out)
