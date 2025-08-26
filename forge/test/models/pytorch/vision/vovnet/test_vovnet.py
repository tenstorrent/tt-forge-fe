# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.vovnet.pytorch import ModelLoader, ModelVariant

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
    ModelVariant.VOVNET27S,
    ModelVariant.VOVNET39,
    ModelVariant.VOVNET57,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vovnet_osmr_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VOVNET,
        variant=variant,
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

    verify_cfg = VerifyConfig()
    verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))

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
        verify_cfg=verify_cfg,
    )

    # Run model on sample data and print results
    loader.print_cls_results(co_out)


@pytest.mark.nightly
def test_vovnet_v1_39_stigma_pytorch():

    variant = ModelVariant.VOVNET39

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VOVNETV1,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
    )

    # Load model and inputs via shared loader
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

    verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Run model on sample data and print results
    loader.print_cls_results(co_out)


@pytest.mark.nightly
def test_vovnet_v1_57_stigma_pytorch():

    variant = ModelVariant.VOVNET57

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VOVNET,
        variant=variant,
        source=Source.OSMR,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and inputs via shared loader
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

    verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Run model on sample data and print results
    loader.print_cls_results(co_out)


variants = [
    ModelVariant.TIMM_VOVNET19B_DW,
    ModelVariant.TIMM_VOVNET39B,
    ModelVariant.TIMM_VOVNET99B,
    ModelVariant.TIMM_VOVNET19B_DW_RAIN1K,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vovnet_timm_pytorch(variant):

    if variant == ModelVariant.TIMM_VOVNET19B_DW_RAIN1K:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VOVNET,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
        group=group,
        priority=priority,
    )

    # Load model and inputs via shared loader
    loader = ModelLoader(variant=ModelVariant(variant))
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
    if variant == ModelVariant.TIMM_VOVNET99B:
        pcc = 0.98

    # Model Verification
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Run model on sample data and print results
    loader.print_cls_results(co_out)
