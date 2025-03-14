# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.mobilenet.utils.utils import (
    load_mobilenet_model,
    post_processing,
)
from test.models.pytorch.vision.utils.utils import load_timm_model_and_input
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.push
def test_mobilenetv1_basic(forge_property_recorder):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mobilenet_v1",
        variant="basic",
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load the model and prepare input data
    framework_model, inputs = load_mobilenet_model("mobilenet_v1")

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    #  Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Inference
    output = compiled_model(*inputs)

    # Post processing
    post_processing(output)


def generate_model_mobilenetv1_imgcls_hf_pytorch(variant):
    # Create Forge module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)
    # tt_model = forge.PyTorchModule("mobilenet_v1__hf_075_192", model)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")

    image_tensor = inputs.pixel_values

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["google/mobilenet_v1_0.75_192"])
def test_mobilenetv1_192(forge_property_recorder, variant):
    pytest.skip("Hitting segmentation fault in MLIR")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mobilnet_v1",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, inputs, _ = generate_model_mobilenetv1_imgcls_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def generate_model_mobilenetV1I224_imgcls_hf_pytorch(variant):
    # Create Forge module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")

    image_tensor = inputs.pixel_values

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["google/mobilenet_v1_1.0_224"])
def test_mobilenetv1_224(forge_property_recorder, variant):
    pytest.skip("Hitting segmentation fault in MLIR")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mobilnet_v1",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model, inputs, _ = generate_model_mobilenetV1I224_imgcls_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


variants = ["mobilenetv1_100.ra4_e3600_r224_in1k"]


@pytest.mark.nightly
@pytest.mark.xfail(reason="RuntimeError: Boolean value of Tensor with more than one value is ambiguous")
@pytest.mark.parametrize("variant", variants)
def test_mobilenet_v1_timm(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mobilenet_v1",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load the model and inputs
    framework_model, inputs = load_timm_model_and_input(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification and Inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
