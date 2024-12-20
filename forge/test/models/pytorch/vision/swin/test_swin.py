# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# STEP 0: import Forge library
import os
import pytest
import timm
from PIL import Image
import requests
from transformers import ViTImageProcessor, Swinv2Model, Swinv2ForImageClassification, Swinv2ForMaskedImageModeling
from test.utils import download_model
import forge
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_swin_v1_tiny_4_224_hf_pytorch(test_device):
    # pytest.skip() # Working on it
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    feature_extractor = ViTImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    # model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", torchscript=True)
    model = download_model(timm.create_model, "swin_tiny_patch4_window7_224", pretrained=True)
    model.eval()

    # STEP 3: Prepare input samples
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # STEP 4: Run inference on Tenstorrent device
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    print(img_tensor.shape)

    inputs = [img_tensor]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_swin_tiny_patch4_window7_224")


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen")
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_4_256_hf_pytorch(variant, test_device):

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2Model.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    inputs = [img_tensor]

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen")
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_image_classification(variant, test_device):

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForImageClassification.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    inputs = [img_tensor]

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen")
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_masked(variant, test_device):

    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    framework_model = Swinv2ForMaskedImageModeling.from_pretrained(variant)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    img_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    inputs = [img_tensor]

    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )
    verify(inputs, framework_model, compiled_model)
