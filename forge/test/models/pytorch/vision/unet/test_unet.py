# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch
from loguru import logger
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from third_party.tt_forge_models.tools.utils import get_file
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
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.unet.model_utils.model import UNET
from test.utils import download_model


def generate_model_unet_imgseg_osmr_pytorch(variant):
    # Also, golden test segfaults when pushing params to golden: tenstorrent/forge#637

    model = download_model(ptcv_get_model, variant, pretrained=False)

    img_tensor = x = torch.randn(1, 3, 224, 224)

    return model.to(torch.bfloat16), [img_tensor.to(torch.bfloat16)], {}


@pytest.mark.xfail
@pytest.mark.nightly
def test_unet_osmr_cityscape_pytorch():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.UNET,
        variant="cityscape",
        source=Source.OSMR,
        task=Task.IMAGE_SEGMENTATION,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    framework_model, inputs, _ = generate_model_unet_imgseg_osmr_pytorch("unet_cityscapes")

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


def get_imagenet_sample():
    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        img = Image.open(file_path).convert("RGB")

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

    return model.to(torch.bfloat16), [img_tensor.to(torch.bfloat16)], {}


@pytest.mark.nightly
@pytest.mark.xfail
def test_unet_qubvel_pytorch():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.UNET,
        variant="qubvel",
        source=Source.TORCH_HUB,
        task=Task.IMAGE_SEGMENTATION,
    )

    framework_model, inputs, _ = generate_model_unet_imgseg_smp_pytorch(None)

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
    file_path = get_file(
        "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
    )
    input_image = Image.open(file_path)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
        ]
    )
    input_tensor = preprocess(input_image)
    img_batch = input_tensor.unsqueeze(0)

    return model.to(torch.bfloat16), [img_batch.to(torch.bfloat16)], {}


@pytest.mark.nightly
@pytest.mark.xfail
def test_unet_torchhub_pytorch():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.UNET, source=Source.TORCH_HUB, task=Task.IMAGE_SEGMENTATION
    )

    framework_model, inputs, _ = generate_model_unet_imgseg_torchhub_pytorch(
        "unet",
    )

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


# Reference: https://github.com/arief25ramadhan/carvana-unet-segmentation
@pytest.mark.nightly
def test_unet_carvana():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.UNETCARVANA,
        source=Source.GITHUB,
        task=Task.IMAGE_SEGMENTATION,
    )

    # Load model and input
    framework_model = UNET(in_channels=3, out_channels=1).to(torch.bfloat16)
    framework_model.eval()
    inputs = [torch.rand((1, 3, 224, 224)).to(torch.bfloat16)]

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
