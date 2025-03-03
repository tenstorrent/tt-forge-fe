# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import urllib

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch
from loguru import logger
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    ConvertImageDtype,
    Normalize,
    PILToTensor,
    Resize,
)

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


def generate_model_unet_imgseg_osmr_pytorch(variant):
    # Also, golden test segfaults when pushing params to golden: tenstorrent/forge#637

    model = download_model(ptcv_get_model, variant, pretrained=False)

    img_tensor = x = torch.randn(1, 3, 224, 224)

    return model, [img_tensor], {}


@pytest.mark.nightly
def test_unet_osmr_cityscape_pytorch(record_forge_property):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="unet", variant="cityscape", source=Source.OSMR, task=Task.IMAGE_SEGMENTATION
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    framework_model, inputs, _ = generate_model_unet_imgseg_osmr_pytorch("unet_cityscapes")

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def get_imagenet_sample():
    try:
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert("RGB")

        # Preprocessing
        transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                PILToTensor(),
                ConvertImageDtype(torch.float32),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Preprocessing
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)
    return img_tensor


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Model script not found")
@pytest.mark.nightly
def test_unet_holocron_pytorch(record_forge_property):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="unet",
        variant="holocron",
        source=Source.TORCH_HUB,
        task=Task.IMAGE_SEGMENTATION,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    from holocron.models.segmentation.unet import unet_tvvgg11

    framework_model = download_model(unet_tvvgg11, pretrained=True).eval()

    img_tensor = get_imagenet_sample()
    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_unet_imgseg_smp_pytorch(variant):
    # encoder_name = "vgg19"
    encoder_name = "resnet101"
    # encoder_name = "vgg19_bn"

    model = download_model(
        smp.Unet,
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    model.eval()

    # Image preprocessing
    params = download_model(smp.encoders.get_preprocessing_params, encoder_name)
    std = torch.tensor(params["std"]).view(1, 3, 1, 1)
    mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)

    image = get_imagenet_sample()
    img_tensor = torch.tensor(image)
    img_tensor = (img_tensor - mean) / std
    print(img_tensor.shape)

    return model, [img_tensor], {}


@pytest.mark.nightly
def test_unet_qubvel_pytorch(record_forge_property):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="unet",
        variant="qubvel",
        source=Source.TORCH_HUB,
        task=Task.IMAGE_SEGMENTATION,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    framework_model, inputs, _ = generate_model_unet_imgseg_smp_pytorch(None)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_unet_imgseg_torchhub_pytorch(variant):
    model = download_model(
        torch.hub.load,
        "mateuszbuda/brain-segmentation-pytorch",
        variant,
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    model.eval()

    # Download an example input image
    url, filename = (
        "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
        "TCGA_CS_4944.png",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
        ]
    )
    input_tensor = preprocess(input_image)
    img_batch = input_tensor.unsqueeze(0)

    return model, [img_batch], {}


@pytest.mark.nightly
def test_unet_torchhub_pytorch(record_forge_property):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="unet", source=Source.TORCH_HUB, task=Task.IMAGE_SEGMENTATION
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    framework_model, inputs, _ = generate_model_unet_imgseg_torchhub_pytorch(
        "unet",
    )

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
