# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

from test.utils import download_model
from test.models.pytorch.vision.resnext.utils.image_utils import get_image_tensor

import forge
from test.models.utils import build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_50_torchhub_pytorch(record_forge_property):
    variant = "resnext50_32x4d"

    module_name = build_module_name(framework="pt", model="resnext", source="torchhub", variant=variant)

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", variant, pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_101_torchhub_pytorch(record_forge_property):
    variant = "resnext101_32x8d"

    module_name = build_module_name(framework="pt", model="resnext", source="torchhub", variant=variant)

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", variant, pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_101_32x8d_fb_wsl_pytorch(record_forge_property):
    variant = "resnext101_32x8d_wsl"

    module_name = build_module_name(framework="pt", model="resnext", source="torchhub", variant=variant)

    # STEP 2: Create Forge module from PyTorch model
    # 4 variants
    model = download_model(torch.hub.load, "facebookresearch/WSL-Images", variant)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_14_osmr_pytorch(record_forge_property):
    variant = "resnext14_32x4d"

    module_name = build_module_name(framework="pt", model="resnext", source="osmr", variant=variant)

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, variant, pretrained=True)
    model.eval()
    # tt_model = forge.PyTorchModule("pt_resnext14_osmr", model)

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_26_osmr_pytorch(record_forge_property):
    variant = "resnext14_32x4d"

    module_name = build_module_name(framework="pt", model="resnext", source="osmr", variant=variant)

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, variant, pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_50_osmr_pytorch(record_forge_property):
    variant = "resnext50_32x4d"

    module_name = build_module_name(framework="pt", model="resnext", source="osmr", variant=variant)

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, variant, pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_101_osmr_pytorch(record_forge_property):
    variant = "resnext101_64x4d"

    module_name = build_module_name(framework="pt", model="resnext", source="osmr", variant=variant)

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, variant, pretrained=True)
    model.eval()

    input_batch = get_image_tensor()

    # STEP 3: Run inference on Tenstorrent device
    # CPU version commented out
    # output = model(input_batch)
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)
