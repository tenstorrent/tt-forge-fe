# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# STEP 0: import Forge library
import pytest
import torch
from third_party.tt_forge_models.swin.pytorch.loader import ModelLoader, ModelVariant
from transformers import (
    SwinForImageClassification,
    Swinv2ForImageClassification,
    Swinv2ForMaskedImageModeling,
    Swinv2Model,
    ViTImageProcessor,
)

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

from test.models.pytorch.vision.swin.model_utils.image_utils import load_image


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/swin-tiny-patch4-window7-224",
        ),
    ],
)
def test_swin_v1_tiny_4_224_hf_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SWIN,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    pytest.xfail(reason="Segmentation fault")

    # STEP 1: Create Forge module from PyTorch model
    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = SwinForImageClassification.from_pretrained(variant).to(torch.bfloat16)
    framework_model.eval()

    # STEP 2: Prepare input samples
    inputs = load_image(feature_extractor)
    inputs = [inputs[0].to(torch.bfloat16)]

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


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/swinv2-tiny-patch4-window8-256",
        ),
    ],
)
def test_swin_v2_tiny_4_256_hf_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SWIN,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Segmentation fault")

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2Model.from_pretrained(variant)

    inputs = load_image(feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/swinv2-tiny-patch4-window8-256",
        ),
    ],
)
def test_swin_v2_tiny_image_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SWIN,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Segmentation Fault")

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForImageClassification.from_pretrained(variant)

    inputs = load_image(feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_masked(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SWIN,
        variant=variant,
        task=Task.MASKED_IMAGE_MODELING,
        source=Source.HUGGINGFACE,
    )

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForMaskedImageModeling.from_pretrained(variant)

    inputs = load_image(feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    ModelVariant.SWIN_T,
    ModelVariant.SWIN_S,
    ModelVariant.SWIN_B,
    pytest.param(ModelVariant.SWIN_V2_T),
    pytest.param(ModelVariant.SWIN_V2_S),
    pytest.param(ModelVariant.SWIN_V2_B),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_swin_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SWIN,
        variant=variant.value,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input using ModelLoader
    loader = ModelLoader(variant=variant)

    if variant in [ModelVariant.SWIN_T, ModelVariant.SWIN_S, ModelVariant.SWIN_B]:
        # Load model and inputs with bfloat16 for Swin v1 variants
        framework_model = loader.load_model(dtype_override=torch.bfloat16)
        inputs = loader.load_inputs(dtype_override=torch.bfloat16)
        inputs = [inputs]  # Wrap in list to match expected format

        data_format_override = DataFormat.Float16_b
        compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    else:
        # Load model and inputs with default dtype for Swin v2 variants
        framework_model = loader.load_model()
        inputs = loader.load_inputs()
        inputs = [inputs]  # Wrap in list to match expected format

        compiler_cfg = CompilerConfig()

    pcc = 0.99

    if variant in [ModelVariant.SWIN_T, ModelVariant.SWIN_V2_T, ModelVariant.SWIN_V2_S]:
        pcc = 0.95
    elif variant == ModelVariant.SWIN_S:
        pcc = 0.92
    elif variant == ModelVariant.SWIN_B:
        pcc = 0.93

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post processing
    loader.print_cls_results(co_out)
