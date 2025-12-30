# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.vgg.pytorch import ModelLoader, ModelVariant

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

# OSMR (pytorchcv) variants
variants = [
    pytest.param(ModelVariant.VGG11),
    pytest.param(ModelVariant.VGG13),
    pytest.param(ModelVariant.VGG16),
    pytest.param(ModelVariant.VGG19),
    pytest.param(ModelVariant.VGG19_BN_OSMR),
    pytest.param(ModelVariant.VGG19_BNB_OSMR),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vgg_osmr_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant=variant.value,
        source=Source.OSMR,
        task=Task.CV_OBJECT_DETECTION,
    )

    # Load model and inputs via loader
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    pcc = 0.99
    if variant in [ModelVariant.VGG16, ModelVariant.VGG19_BN_OSMR]:
        pcc = 0.98
    elif variant == ModelVariant.VGG19:
        pcc = 0.97
    elif variant == ModelVariant.VGG19_BNB_OSMR:
        pcc = 0.96

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


@pytest.mark.nightly
def test_vgg_19_hf_pytorch():

    variant = ModelVariant.HF_VGG19

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant="19",
        source=Source.HUGGINGFACE,
        task=Task.CV_OBJECT_DETECTION,
    )

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Run model on sample data and print results
    loader.print_cls_results(co_out)


@pytest.mark.nightly
def test_vgg_bn19_timm_pytorch():

    variant = ModelVariant.TIMM_VGG19_BN

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant="vgg19_bn",
        source=Source.TIMM,
        task=Task.CV_OBJECT_DETECTION,
    )

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Run model on sample data and print results
    loader.print_cls_results(co_out)


@pytest.mark.nightly
def test_vgg_bn19_torchhub_pytorch():

    variant = ModelVariant.VGG19_BN

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.CV_OBJECT_DETECTION,
    )

    # Load model and inputs via loader
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
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    loader.print_cls_results(co_out)


variants = [
    ModelVariant.TV_VGG11,
    ModelVariant.TV_VGG11_BN,
    ModelVariant.TV_VGG13,
    ModelVariant.TV_VGG13_BN,
    ModelVariant.TV_VGG16,
    ModelVariant.TV_VGG16_BN,
    ModelVariant.TV_VGG19,
    ModelVariant.TV_VGG19_BN,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vgg_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
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
    if variant in [ModelVariant.TV_VGG16_BN, ModelVariant.TV_VGG13_BN]:
        pcc = 0.98

    # Model Verification and inference
    _, co_out = verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    # Run model on sample data and print results
    loader.print_cls_results(co_out)
