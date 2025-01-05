# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge

import torch
import torchxrayvision as xrv
from test.models.pytorch.vision.densenet.utils.densenet_utils import get_input_img, get_input_img_hf_xray
from test.models.utils import build_module_name, Framework


variants = ["densenet121", "densenet121_hf_xray"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_densenet_121_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="densenet121", variant=variant)

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    if variant == "densenet121":
        model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet121", pretrained=True)
        img_tensor = get_input_img()
    else:
        model_name = "densenet121-res224-all"
        model = download_model(xrv.models.get_model, model_name)
        img_tensor = get_input_img_hf_xray()

    # STEP 3: Run inference on Tenstorrent device
    model(img_tensor)
    inputs = [img_tensor]
    variant_name = variant.replace("-", "_")
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_densenet_161_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="densenet161")

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet161", pretrained=True)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()
    model(img_tensor)
    inputs = [img_tensor]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_densenet_169_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="densenet169")

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet169", pretrained=True)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()
    model(img_tensor)
    inputs = [img_tensor]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_densenet_201_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="densenet201")

    record_forge_property("module_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet201", pretrained=True)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()
    model(img_tensor)
    inputs = [img_tensor]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
