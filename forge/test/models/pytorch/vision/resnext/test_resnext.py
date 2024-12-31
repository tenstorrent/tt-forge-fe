# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

from test.utils import download_model
from test.models.pytorch.vision.resnext.utils.image_utils import get_image_tensor

import forge
from forge.test.models.utils import build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_50_torchhub_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "resnext50_32x4d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    module_name = build_module_name(framework="pt", model="resnext_torchhub", variant="50")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_101_torchhub_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "resnext101_32x8d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    module_name = build_module_name(framework="pt", model="resnext_torchhub", variant="101")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_101_32x8d_fb_wsl_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    # 4 variants
    model = download_model(torch.hub.load, "facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    module_name = build_module_name(framework="pt", model="resnext_fb_wsl", variant="101")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_14_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext14_32x4d", pretrained=True)
    model.eval()
    # tt_model = forge.PyTorchModule("pt_resnext14_osmr", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    module_name = build_module_name(framework="pt", model="resnext_osmr", variant="14")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_26_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext26_32x4d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    module_name = build_module_name(framework="pt", model="resnext_osmr", variant="26")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_50_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext50_32x4d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    module_name = build_module_name(framework="pt", model="resnext_osmr", variant="50")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_101_osmr_pytorch(test_device):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, "resnext101_64x4d", pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    module_name = build_module_name(framework="pt", model="resnext_osmr", variant="101")
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)
