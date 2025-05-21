# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import timm
import torch
from loguru import logger
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    MobileNetV2ForSemanticSegmentation,
)

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, ModelGroup, Source, Task
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.mobilenet.model_utils.utils import (
    load_mobilenet_model,
    post_processing,
)
from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.push
def test_mobilenetv2_basic(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilenetv2",
        variant="basic",
        source=Source.TORCH_HUB,
        task=Task.IMAGE_CLASSIFICATION,
        group=ModelGroup.RED,
    )

    # Load the model and prepare input data
    framework_model, inputs = load_mobilenet_model("mobilenet_v2")

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Post processing
    post_processing(co_out)


def generate_model_mobilenetV2I96_imgcls_hf_pytorch(variant):
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    image_tensor = inputs.pixel_values

    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["google/mobilenet_v2_0.35_96"])
def test_mobilenetv2_96(forge_property_recorder, variant):
    pytest.skip("Hitting segmentation fault in MLIR")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilenetv2",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    framework_model, inputs, _ = generate_model_mobilenetV2I96_imgcls_hf_pytorch(variant)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def generate_model_mobilenetV2I160_imgcls_hf_pytorch(variant):
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")
    image_tensor = inputs.pixel_values

    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["google/mobilenet_v2_0.75_160"])
def test_mobilenetv2_160(forge_property_recorder, variant):
    pytest.skip("Hitting segmentation fault in MLIR")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilenetv2",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    framework_model, inputs, _ = generate_model_mobilenetV2I160_imgcls_hf_pytorch(variant)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def generate_model_mobilenetV2I244_imgcls_hf_pytorch(variant):
    # Create Forge module from PyTorch model
    preprocessor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(AutoModelForImageClassification.from_pretrained, variant)

    # Image load and pre-processing into pixel_values
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = preprocessor(images=image, return_tensors="pt")

    image_tensor = inputs.pixel_values

    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["google/mobilenet_v2_1.0_224"])
def test_mobilenetv2_224(forge_property_recorder, variant):
    pytest.skip("Hitting segmentation fault in MLIR")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilenetv2",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    framework_model, inputs, _ = generate_model_mobilenetV2I244_imgcls_hf_pytorch(variant)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def generate_model_mobilenetV2_imgcls_timm_pytorch(variant):
    model = download_model(timm.create_model, variant, pretrained=True)
    # tt_model = forge.PyTorchModule("mobilenet_v2__hf_timm", model)

    # Image load and pre-processing into pixel_values
    try:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        img = Image.open(
            requests.get("https://github.com/pytorch/hub/raw/master/images/dog.jpg", stream=True).raw
        ).convert("RGB")
        image_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image_tensor = torch.rand(1, 3, 224, 224)

    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["mobilenetv2_100"])
def test_mobilenetv2_timm(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilenetv2",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    framework_model, inputs, _ = generate_model_mobilenetV2_imgcls_timm_pytorch(variant)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


def generate_model_mobilenetV2_semseg_hf_pytorch(variant):
    # This variant with input size 3x224x224 works with manual kernel fracturing
    # of the first op. Pad between input activations and first convolution needs
    # to be hoist to the input in order for pre-striding to work (no need for
    # manual kernel fracturing).

    # Load model
    framework_model = download_model(MobileNetV2ForSemanticSegmentation.from_pretrained, variant)

    try:
        config = resolve_data_config({}, model=framework_model)
        transform = create_transform(**config)
        img = Image.open(
            requests.get("https://github.com/pytorch/hub/raw/master/images/dog.jpg", stream=True).raw
        ).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    return framework_model.to(torch.bfloat16), [img_tensor.to(torch.bfloat16)], {}


variants = ["google/deeplabv3_mobilenet_v2_1.0_513"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_mobilenetv2_deeplabv3(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilnetv2",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    framework_model, inputs, _ = generate_model_mobilenetV2_semseg_hf_pytorch(variant)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


variants_with_weights = {
    "mobilenet_v2": "MobileNet_V2_Weights",
}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_mobilenetv2_torchvision(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilenetv2",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)
    framework_model = framework_model.to(torch.bfloat16)
    inputs = inputs.to(torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
