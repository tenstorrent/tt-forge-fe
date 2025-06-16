# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# STEP 0: import Forge library
import pytest
import torch
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
from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/swin-tiny-patch4-window7-224",
            marks=[pytest.mark.skip(reason="Transient failure - Segmentation fault")],
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

    # STEP 1: Create Forge module from PyTorch model
    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = SwinForImageClassification.from_pretrained(variant).to(torch.bfloat16)
    framework_model.eval()

    # STEP 2: Prepare input samples
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)
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
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/swinv2-tiny-patch4-window8-256",
            marks=pytest.mark.xfail,
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

    raise RuntimeError("Transient failure - Segmentation fault")

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2Model.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/swinv2-tiny-patch4-window8-256",
            marks=[pytest.mark.skip(reason="Transient failure - Segmentation fault")],
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

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForImageClassification.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

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

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants_with_weights = {
    "swin_t": "Swin_T_Weights",
    "swin_s": "Swin_S_Weights",
    "swin_b": "Swin_B_Weights",
    "swin_v2_t": "Swin_V2_T_Weights",
    "swin_v2_s": "Swin_V2_S_Weights",
    "swin_v2_b": "Swin_V2_B_Weights",
}

variants = [
    "swin_t",
    "swin_s",
    "swin_b",
    pytest.param("swin_v2_t", marks=[pytest.mark.xfail]),
    pytest.param("swin_v2_s", marks=[pytest.mark.xfail]),
    pytest.param("swin_v2_b", marks=[pytest.mark.xfail]),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_swin_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SWIN,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)

    if variant in ["swin_t", "swin_s", "swin_b"]:

        framework_model.to(torch.bfloat16)
        inputs = [inputs[0].to(torch.bfloat16)]

        data_format_override = DataFormat.Float16_b
        compiler_cfg = CompilerConfig(default_df_override=data_format_override, enable_optimization_passes=True)

    else:
        compiler_cfg = CompilerConfig(enable_optimization_passes=True)

    pcc = 0.99

    if variant == "swin_t":
        pcc = 0.97
    elif variant == "swin_s":
        pcc = 0.92
    elif variant == "swin_b":
        pcc = 0.93

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
