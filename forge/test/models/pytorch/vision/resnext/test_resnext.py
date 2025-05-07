# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.resnext.utils.utils import (
    get_image_tensor,
    get_resnext_model_and_input,
    post_processing,
)
from test.utils import download_model


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext50_32x4d"])
def test_resnext_50_torchhub_pytorch(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="resnext",
        source=Source.TORCH_HUB,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model and prepare input data
    framework_model, inputs = get_resnext_model_and_input("pytorch/vision:v0.10.0", variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Post processing
    post_processing(co_out)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext101_32x8d"])
def test_resnext_101_torchhub_pytorch(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="resnext",
        source=Source.TORCH_HUB,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model and prepare input data
    framework_model, inputs = get_resnext_model_and_input("pytorch/vision:v0.10.0", variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Post processing
    post_processing(co_out)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext101_32x8d_wsl"])
def test_resnext_101_32x8d_fb_wsl_pytorch(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="resnext",
        source=Source.TORCH_HUB,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model and prepare input data
    framework_model = download_model(torch.hub.load, "facebookresearch/WSL-Images", variant)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext14_32x4d"])
def test_resnext_14_osmr_pytorch(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="resnext",
        source=Source.OSMR,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model and prepare input data
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext26_32x4d"])
def test_resnext_26_osmr_pytorch(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="resnext",
        source=Source.OSMR,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext50_32x4d"])
def test_resnext_50_osmr_pytorch(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="resnext",
        source=Source.OSMR,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext101_64x4d"])
def test_resnext_101_osmr_pytorch(forge_property_recorder, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="resnext",
        source=Source.OSMR,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
