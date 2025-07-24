# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

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
from forge.verify.verify import verify

from test.models.pytorch.vision.mobilenet.model_utils.utils import (
    load_mobilenet_model,
    post_processing,
)
from test.models.pytorch.vision.vision_utils.utils import load_timm_model_and_input
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.push
def test_mobilenetv1_basic():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MOBILENETV1,
        variant="basic",
        source=Source.TORCHVISION,
        task=Task.CV_IMAGE_CLS,
    )

    # Load the model and prepare input data
    framework_model, inputs = load_mobilenet_model("mobilenet_v1")
    framework_model.to(torch.bfloat16)
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

    #  Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    post_processing(co_out)


def generate_model_mobilenetv1_imgcls_hf_pytorch(variant):
    # Create Forge module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)
    # tt_model = forge.PyTorchModule("mobilenet_v1__hf_075_192", model)

    # Image load and pre-processing into pixel_values
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    image = next(iter(dataset.skip(10)))["image"]
    inputs = preprocessor(images=image, return_tensors="pt")

    image_tensor = inputs.pixel_values

    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["google/mobilenet_v1_0.75_192"])
def test_mobilenetv1_192(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MOBILENETV1,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLS,
    )
    pytest.xfail(reason="Hitting segmentation fault in MLIR")

    framework_model, inputs, _ = generate_model_mobilenetv1_imgcls_hf_pytorch(variant)

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


def generate_model_mobilenetV1I224_imgcls_hf_pytorch(variant):
    # Create Forge module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    image = next(iter(dataset.skip(10)))["image"]
    inputs = preprocessor(images=image, return_tensors="pt")

    image_tensor = inputs.pixel_values

    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["google/mobilenet_v1_1.0_224"])
def test_mobilenetv1_224(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MOBILENETV1,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.CV_IMAGE_CLS,
    )
    pytest.xfail(reason="Hitting segmentation fault in MLIR")

    framework_model, inputs, _ = generate_model_mobilenetV1I224_imgcls_hf_pytorch(variant)

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


variants = ["mobilenetv1_100.ra4_e3600_r224_in1k"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_mobilenet_v1_timm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MOBILENETV1,
        variant=variant,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLS,
    )

    # Load the model and inputs
    framework_model, inputs = load_timm_model_and_input(variant)
    framework_model = framework_model.to(torch.bfloat16)
    inputs = [inp.to(torch.bfloat16) for inp in inputs]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    verify(inputs, framework_model, compiled_model)
