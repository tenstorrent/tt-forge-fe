# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# STEP 0: import Forge library
import os
import pytest
import timm
from transformers import ViTImageProcessor, Swinv2Model, Swinv2ForImageClassification, Swinv2ForMaskedImageModeling
from test.utils import download_model
import forge
from forge.verify.verify import verify
from test.models.pytorch.vision.swin.utils.image_utils import load_image
from forge.test.models.utils import build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(
    reason='RuntimeError: TT_ASSERT @ forge/csrc/passes/commute_utils.cpp:1105: reshape->op_name() == "reshape"'
)
@pytest.mark.parametrize("variant", ["microsoft/swin-tiny-patch4-window7-224"])
def test_swin_v1_tiny_4_224_hf_pytorch(variant):

    # STEP 1: Create Forge module from PyTorch model
    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    # model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", torchscript=True)
    model = download_model(timm.create_model, variant, pretrained=True)
    model.eval()

    # STEP 2: Prepare input samples
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # STEP 3: Run inference on Tenstorrent device
    module_name = build_module_name(framework="pt", model="swin", variant=variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
    verify(inputs, model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen")
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_4_256_hf_pytorch(variant):

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2Model.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    module_name = build_module_name(framework="pt", model="swin", variant=variant)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen")
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_image_classification(variant):

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForImageClassification.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    module_name = build_module_name(framework="pt", model="swin", variant=variant, task="image_classification")
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen")
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_masked(variant):

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForMaskedImageModeling.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    module_name = build_module_name(framework="pt", model="swin", variant=variant, task="masked")
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)
    verify(inputs, framework_model, compiled_model)
