# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from datasets import load_dataset
from loguru import logger
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms

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

from test.models.models_utils import print_cls_results
from test.utils import download_model


@pytest.mark.nightly
def test_alexnet_torchhub():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ALEXNET,
        source=Source.TORCH_HUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "alexnet", pretrained=True).to(
        torch.bfloat16
    )
    framework_model.eval()

    # Load and pre-process image
    try:

        dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
        input_image = next(iter(dataset.skip(10)))["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(input_image).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    inputs = [img_tensor.to(torch.bfloat16)]

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
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    print_cls_results(fw_out[0], co_out[0])


@pytest.mark.nightly
def test_alexnet_osmr():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.ALEXNET, source=Source.OSMR, task=Task.IMAGE_CLASSIFICATION
    )

    # Load model

    # Using AlexNet-b instead of AlexNet-a to avoid LocalResponseNorm,
    # which internally uses avgpool3d — currently unsupported for bfloat16.
    framework_model = download_model(ptcv_get_model, "alexnetb", pretrained=True).to(torch.bfloat16)
    framework_model.eval()

    # Load and pre-process image
    try:
        dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
        input_image = next(iter(dataset.skip(10)))["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(input_image).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    inputs = [img_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    print_cls_results(fw_out[0], co_out[0])
