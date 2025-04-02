# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# STEP 0: import Forge library
import pytest
from transformers import (
    SwinForImageClassification,
    Swinv2ForImageClassification,
    Swinv2ForMaskedImageModeling,
    Swinv2Model,
    ViTImageProcessor,
)

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.swin.utils.image_utils import load_image
from test.models.pytorch.vision.utils.utils import load_vision_model_and_input
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/swin-tiny-patch4-window7-224",
            marks=[pytest.mark.xfail],
        ),
    ],
)
def test_swin_v1_tiny_4_224_hf_pytorch(forge_property_recorder, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="swin",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # STEP 1: Create Forge module from PyTorch model
    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = SwinForImageClassification.from_pretrained(variant)
    framework_model.eval()

    # STEP 2: Prepare input samples
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/swinv2-tiny-patch4-window8-256",
            marks=[pytest.mark.xfail],
        ),
    ],
)
def test_swin_v2_tiny_4_256_hf_pytorch(forge_property_recorder, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="swin",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_model_name(module_name)

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2Model.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_image_classification(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="swin",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForImageClassification.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_masked(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="swin",
        variant=variant,
        task=Task.MASKED_IMAGE_MODELLING,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForMaskedImageModeling.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


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
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_swin_torchvision(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="swin",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
