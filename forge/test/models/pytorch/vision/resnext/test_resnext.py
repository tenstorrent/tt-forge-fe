# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

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

from test.models.pytorch.vision.resnext.model_utils.utils import (
    get_image_tensor,
    get_resnext_model_and_input,
    post_processing,
)
from test.utils import download_model


# @pytest.mark.push
# @pytest.mark.nightly
# @pytest.mark.parametrize("variant", ["resnext50_32x4d"])
# def test_resnext_50_torchhub_pytorch(variant):
#     # Record Forge Property
#     module_name = record_model_properties(
#         framework=Framework.PYTORCH,
#         model=ModelArch.RESNEXT,
#         source=Source.TORCH_HUB,
#         variant=variant,
#         task=Task.IMAGE_CLASSIFICATION,
#     )
#
#     # Load the model and prepare input data
#     framework_model, inputs = get_resnext_model_and_input("pytorch/vision:v0.10.0", variant)
#     framework_model.to(torch.bfloat16)
#     inputs = [inputs[0].to(torch.bfloat16)]
#
#     data_format_override = DataFormat.Float16_b
#     compiler_cfg = CompilerConfig(default_df_override=data_format_override)
#
#     # Forge compile framework model
#     compiled_model = forge.compile(
#         framework_model,
#         sample_inputs=inputs,
#         module_name=module_name,
#         compiler_cfg=compiler_cfg,
#     )
#
#     # Model Verification and Inference
#     _, co_out = verify(inputs, framework_model, compiled_model)
#
#     # Post processing
#     post_processing(co_out)


# @pytest.mark.push
# @pytest.mark.nightly
# @pytest.mark.parametrize("variant", ["resnext101_32x8d"])
# def test_resnext_101_torchhub_pytorch(variant):
#     # Record Forge Property
#     module_name = record_model_properties(
#         framework=Framework.PYTORCH,
#         model=ModelArch.RESNEXT,
#         source=Source.TORCH_HUB,
#         variant=variant,
#         task=Task.IMAGE_CLASSIFICATION,
#     )
#
#     # Load the model and prepare input data
#     framework_model, inputs = get_resnext_model_and_input("pytorch/vision:v0.10.0", variant)
#     framework_model.to(torch.bfloat16)
#     inputs = [inputs[0].to(torch.bfloat16)]
#
#     data_format_override = DataFormat.Float16_b
#     compiler_cfg = CompilerConfig(default_df_override=data_format_override)
#
#     # Forge compile framework model
#     compiled_model = forge.compile(
#         framework_model,
#         sample_inputs=inputs,
#         module_name=module_name,
#         compiler_cfg=compiler_cfg,
#     )
#
#     # Model Verification and Inference
#     _, co_out = verify(inputs, framework_model, compiled_model)
#
#     # Post processing
#     post_processing(co_out)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext101_32x8d_wsl"])
def test_resnext_101_32x8d_fb_wsl_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNEXT,
        source=Source.TORCH_HUB,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and prepare input data
    framework_model = download_model(torch.hub.load, "facebookresearch/WSL-Images", variant)
    framework_model.eval()
    framework_model.to(torch.bfloat16)

    input_batch = get_image_tensor()
    inputs = [input_batch.to(torch.bfloat16)]

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    post_processing(co_out)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext14_32x4d"])
def test_resnext_14_osmr_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNEXT,
        source=Source.OSMR,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and prepare input data
    framework_model = download_model(ptcv_get_model, variant, pretrained=True).to(torch.bfloat16)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch.to(torch.bfloat16)]

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    post_processing(co_out)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext26_32x4d"])
def test_resnext_26_osmr_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNEXT,
        source=Source.OSMR,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True).to(torch.bfloat16)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch.to(torch.bfloat16)]

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    post_processing(co_out)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext50_32x4d"])
def test_resnext_50_osmr_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNEXT,
        source=Source.OSMR,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True).to(torch.bfloat16)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch.to(torch.bfloat16)]

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    post_processing(co_out)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["resnext101_64x4d"])
def test_resnext_101_osmr_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNEXT,
        source=Source.OSMR,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True).to(torch.bfloat16)
    framework_model.eval()

    input_batch = get_image_tensor()
    inputs = [input_batch.to(torch.bfloat16)]

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    post_processing(co_out)
