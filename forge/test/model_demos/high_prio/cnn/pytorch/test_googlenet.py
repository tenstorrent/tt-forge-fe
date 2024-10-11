# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from forge.verify.backend import verify_module
from forge import VerifyConfig
from forge._C.backend_api import BackendType, BackendDevice
from forge.verify.config import TestKind, NebulaGalaxy

import os

import forge
import torch
from PIL import Image
from torchvision import models, transforms
from loguru import logger


def test_googlenet_pytorch(test_device):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b

    # Create Forge module from PyTorch model
    # Two ways to load the same model
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model = download_model(models.googlenet, pretrained=True)
    model.eval()
    tt_model = forge.PyTorchModule("pt_googlenet", model)

    # Image preprocessing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids
            if "FORGE_NEB_GALAXY_CI" in os.environ and int(os.environ.get("FORGE_NEB_GALAXY_CI")) == 1
            else [0],
        ),
    )
