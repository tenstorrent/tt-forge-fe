# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.resnext.utils.image_utils import get_image_tensor
from test.models.utils import Framework, Source, build_module_name
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_50_torchhub_pytorch(record_forge_property):
    variant = "resnext50_32x4d"

    module_name = build_module_name(
        framework=Framework.PYTORCH, model="resnext", source=Source.TORCH_HUB, variant=variant
    )

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_101_torchhub_pytorch(record_forge_property):
    variant = "resnext101_32x8d"

    module_name = build_module_name(
        framework=Framework.PYTORCH, model="resnext", source=Source.TORCH_HUB, variant=variant
    )

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_101_32x8d_fb_wsl_pytorch(record_forge_property):
    variant = "resnext101_32x8d_wsl"

    module_name = build_module_name(
        framework=Framework.PYTORCH, model="resnext", source=Source.TORCH_HUB, variant=variant
    )

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    # 4 variants
    framework_model = download_model(torch.hub.load, "facebookresearch/WSL-Images", variant)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_14_osmr_pytorch(record_forge_property):
    variant = "resnext14_32x4d"

    module_name = build_module_name(framework=Framework.PYTORCH, model="resnext", source=Source.OSMR, variant=variant)

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_26_osmr_pytorch(record_forge_property):
    variant = "resnext14_32x4d"

    module_name = build_module_name(framework=Framework.PYTORCH, model="resnext", source=Source.OSMR, variant=variant)

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_50_osmr_pytorch(record_forge_property):
    variant = "resnext50_32x4d"

    module_name = build_module_name(framework=Framework.PYTORCH, model="resnext", source=Source.OSMR, variant=variant)

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_resnext_101_osmr_pytorch(record_forge_property):
    variant = "resnext101_64x4d"

    module_name = build_module_name(framework=Framework.PYTORCH, model="resnext", source=Source.OSMR, variant=variant)

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
