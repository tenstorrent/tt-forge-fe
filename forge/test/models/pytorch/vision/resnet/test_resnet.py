# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import timm
import torch
from datasets import load_dataset
from loguru import logger
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoImageProcessor, ResNetForImageClassification

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

variants = [
    "microsoft/resnet-50",
]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_resnet_hf(variant, record_forge_property):
    # Record model properties
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="resnet",
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    record_forge_property("model_name", module_name)

    # Load dataset
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    # Load Torch model, preprocess image, and label dictionary
    processor = download_model(AutoImageProcessor.from_pretrained, variant)
    framework_model = download_model(ResNetForImageClassification.from_pretrained, variant, return_dict=False)
    label_dict = framework_model.config.id2label

    inputs = processor(image, return_tensors="pt")
    inputs = inputs["pixel_values"]

    compiled_model = forge.compile(framework_model, inputs)

    cpu_logits = framework_model(inputs)[0]
    cpu_pred = label_dict[cpu_logits.argmax(-1).item()]

    tt_logits = compiled_model(inputs)[0]
    tt_pred = label_dict[tt_logits.argmax(-1).item()]

    assert cpu_pred == tt_pred, f"Inference mismatch: CPU prediction: {cpu_pred}, TT prediction: {tt_pred}"

    verify([inputs], framework_model, compiled_model)


def generate_model_resnet_imgcls_timm_pytorch(variant):
    # Load ResNet50 feature extractor and model from TIMM
    model = download_model(timm.create_model, variant, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # Load data sample
    try:
        url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image = torch.rand(1, 3, 256, 256)

    # Data preprocessing
    pixel_values = transform(image).unsqueeze(0)

    return model, [pixel_values], {}


@pytest.mark.nightly
def test_resnet_timm(record_forge_property):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="resnet", source=Source.TIMM, variant="50", task=Task.IMAGE_CLASSIFICATION
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    framework_model, inputs, _ = generate_model_resnet_imgcls_timm_pytorch("resnet50")

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
