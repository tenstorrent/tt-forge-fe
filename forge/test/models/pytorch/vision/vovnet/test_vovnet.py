# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model

import forge

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from test.models.pytorch.vision.vovnet.utils.src_vovnet_stigma import vovnet39, vovnet57
from test.models.pytorch.vision.vovnet.utils.model_utils import get_image, preprocess_steps, preprocess_timm_model


def generate_model_vovnet_imgcls_osmr_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_compiler_config()  # load compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, variant, pretrained=True)
    image_tensor = get_image()

    return model, [image_tensor], {} , compiler_cfg=compiler_cfg


varaints = ["vovnet27s", "vovnet39", "vovnet57"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", varaints, ids=varaints)
def test_vovnet_osmr_pytorch(variant, test_device):
    model, inputs, _, compiler_cfg = generate_model_vovnet_imgcls_osmr_pytorch(
        test_device,
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=f"pt_{variant}", compiler_cfg=compiler_cfg)


def generate_model_vovnet39_imgcls_stigma_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet39)
    return model, [image_tensor], {}, compiler_cfg


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("enable_default_dram_parameters", [True, False])
def test_vovnet_v1_39_stigma_pytorch(test_device, enable_default_dram_parameters):
    model, inputs, _ , compiler_cfg = generate_model_vovnet39_imgcls_stigma_pytorch(
        test_device,
        None,
    )

    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=f"pt_vovnet_39_stigma", compiler_cfg=compiler_cfg)


def generate_model_vovnet57_imgcls_stigma_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet57)

    return model, [image_tensor], {}, compiler_cfg


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_vovnet_v1_57_stigma_pytorch(test_device):
    model, inputs, _, compiler_cfg = generate_model_vovnet57_imgcls_stigma_pytorch(
        test_device,
        None,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=f"vovnet_57_stigma_pt", compiler_cfg=compiler_cfg)


def generate_model_vovnet_imgcls_timm_pytorch(test_device, variant):
    model, image_tensor = download_model(preprocess_timm_model, variant)
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.CONSTEVAL_GRAPH

    return model, [image_tensor], {}, compiler_cfg


variants = ["ese_vovnet19b_dw", "ese_vovnet39b", "ese_vovnet99b"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vovnet_timm_pytorch(variant, test_device):
    model, inputs, _, compiler_cfg = generate_model_vovnet_imgcls_timm_pytorch(
        test_device,
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=f"pt_{variant}", compiler_cfg=compiler_cfg)
