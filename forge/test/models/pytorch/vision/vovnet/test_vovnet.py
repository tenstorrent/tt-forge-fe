# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model

import forge

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from test.models.pytorch.vision.vovnet.utils.src_vovnet_stigma import vovnet39, vovnet57
from test.models.pytorch.vision.vovnet.utils.model_utils import get_image, preprocess_steps, preprocess_timm_model
from test.models.utils import build_module_name, Framework, Task


def generate_model_vovnet_imgcls_osmr_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    model = download_model(ptcv_get_model, variant, pretrained=True)
    image_tensor = get_image()

    return model, [image_tensor], {}


varaints = ["vovnet27s", "vovnet39", "vovnet57"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", varaints, ids=varaints)
def test_vovnet_osmr_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="vovnet", variant=variant, source="osmr")

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_vovnet_imgcls_osmr_pytorch(
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)


def generate_model_vovnet39_imgcls_stigma_pytorch():
    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet39)
    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_vovnet_v1_39_stigma_pytorch(record_forge_property):
    variant = "vovnet39"

    module_name = build_module_name(framework=Framework.PYTORCH, model="vovnet_v1", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_vovnet39_imgcls_stigma_pytorch()

    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)


def generate_model_vovnet57_imgcls_stigma_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet57)

    return model, [image_tensor], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_vovnet_v1_57_stigma_pytorch(record_forge_property):
    variant = "vovnet_v1_57"

    module_name = build_module_name(framework=Framework.PYTORCH, model="vovnet", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_vovnet57_imgcls_stigma_pytorch()
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)


def generate_model_vovnet_imgcls_timm_pytorch(variant):
    model, image_tensor = download_model(preprocess_timm_model, variant)

    return model, [image_tensor], {}


variants = ["ese_vovnet19b_dw", "ese_vovnet39b", "ese_vovnet99b"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vovnet_timm_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="vovnet", variant=variant)

    record_forge_property("module_name", module_name)

    model, inputs, _ = generate_model_vovnet_imgcls_timm_pytorch(
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]], module_name=module_name)
