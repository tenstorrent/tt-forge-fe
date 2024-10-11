# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from forge.verify.backend import verify_module
from forge import VerifyConfig
from forge._C.backend_api import BackendType, BackendDevice
from forge.verify.config import TestKind

import os
from loguru import logger
import forge
import torch
from PIL import Image
from torchvision import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model


#############
def get_image_tensor():
    # Image processing
    try:
        torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        input_image = Image.open("dog.jpg")
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
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
    return input_batch


def test_resnext_50_torchhub_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["FORGE_RIBBON2"] = "1"
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b

    if test_device.arch == BackendDevice.Grayskull:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{72*1024}"

    os.environ["FORGE_TEMP_DISABLE_MODEL_KB_PROLOGUE_BW"] = "1"

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "resnext50_32x4d", pretrained=True)
    model.eval()
    tt_model = forge.PyTorchModule("pt_resnext50_torchhub", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


def test_resnext_101_torchhub_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["FORGE_RIBBON2"] = "1"
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["FORGE_BALANCER_PREPASS_DISABLED"] = "1"
    elif test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False
        os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "resnext101_32x8d", pretrained=True)
    model.eval()
    tt_model = forge.PyTorchModule("pt_resnext101_torchhub", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


def test_resnext_101_32x8d_fb_wsl_pytorch(test_device):

    if test_device.arch == BackendDevice.Grayskull:
        pytest.skip(
            "Grayskull failing with: <PIPEGEN-ERROR> Chip = 0, Core x = 1, y = 7(logical x = 0, y = 5): has more than 24 prefetch buf streams"
        )

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["FORGE_RIBBON2"] = "1"
    os.environ["FORGE_BALANCER_PREPASS_DISABLED"] = "1"
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b

    # STEP 2: Create Forge module from PyTorch model
    # 4 variants
    model = download_model(torch.hub.load, "facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    model.eval()
    tt_model = forge.PyTorchModule("pt_resnext101_fb_wsl", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


def test_resnext_14_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["FORGE_RIBBON2"] = "1"
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b

    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False
        os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{24*1024}"

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext14_32x4d", pretrained=True)
    model.eval()
    tt_model = forge.PyTorchModule("pt_resnext14_osmr", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
        ),
    )


def test_resnext_26_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    os.environ["FORGE_RIBBON2"] = "1"

    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False
        os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{72*1024}"

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext26_32x4d", pretrained=True)
    model.eval()
    tt_model = forge.PyTorchModule("pt_resnext26_osmr", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
        ),
    )


def test_resnext_50_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["FORGE_RIBBON2"] = "1"
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    if test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False
        os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{72*1024}"

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext50_32x4d", pretrained=True)
    model.eval()
    tt_model = forge.PyTorchModule("pt_resnext50_osmr", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


def test_resnext_101_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["FORGE_RIBBON2"] = "1"
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b

    if test_device.arch == BackendDevice.Wormhole_B0:
        os.environ["FORGE_BALANCER_PREPASS_DISABLED"] = "1"
    elif test_device.arch == BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False
        os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{80*1024}"

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext101_64x4d", pretrained=True)
    model.eval()
    tt_model = forge.PyTorchModule("pt_resnext101_osmr", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)

    verify_module(
        tt_model,
        input_shapes=[(input_batch.shape,)],
        inputs=[(input_batch,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
