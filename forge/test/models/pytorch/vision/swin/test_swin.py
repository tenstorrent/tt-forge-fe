# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# STEP 0: import Forge library
import pytest
import timm
from transformers import (
    Swinv2ForImageClassification,
    Swinv2ForMaskedImageModeling,
    Swinv2Model,
    ViTImageProcessor,
)

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.swin.utils.image_utils import load_image
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["microsoft/swin-tiny-patch4-window7-224"])
def test_swin_v1_tiny_4_224_hf_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="swin",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # STEP 1: Create Forge module from PyTorch model
    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    # model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", torchscript=True)
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 2: Prepare input samples
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_4_256_hf_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="swin",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

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
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_image_classification(record_forge_property, variant):
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
    record_forge_property("model_name", module_name)

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
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_masked(record_forge_property, variant):
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
    record_forge_property("model_name", module_name)

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForMaskedImageModeling.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
