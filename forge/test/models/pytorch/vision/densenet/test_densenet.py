# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn
import torchxrayvision as xrv
from torchxrayvision.models import fix_resolution, op_norm

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.densenet.utils.densenet_utils import (
    get_input_img,
    get_input_img_hf_xray,
)
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

variants = ["densenet121", "densenet121_hf_xray"]


class densenet_xray_wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = fix_resolution(x, 224, self.model)
        features = self.model.features2(x)
        out = self.model.classifier(features)
        out = torch.sigmoid(out)
        return out


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_densenet_121_pytorch(record_forge_property, variant):
    if variant == "densenet121":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="densenet",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    if variant == "densenet121":
        framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet121", pretrained=True)
        img_tensor = get_input_img()
    else:
        model_name = "densenet121-res224-all"
        model = download_model(xrv.models.get_model, model_name)
        framework_model = densenet_xray_wrapper(model)
        img_tensor = get_input_img_hf_xray()

    # STEP 3: Run inference on Tenstorrent device
    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    if variant == "densenet121_hf_xray":
        # Inference
        output = compiled_model(*inputs)
        # post processing
        outputs = op_norm(output[0], model.op_threshs)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["densenet161"])
def test_densenet_161_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="densenet",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet161", pretrained=True)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()
    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["densenet169"])
def test_densenet_169_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="densenet",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet169", pretrained=True)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()

    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["densenet201"])
def test_densenet_201_pytorch(record_forge_property, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a more than 32 GB during compile time)")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="densenet",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet201", pretrained=True)

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()

    inputs = [img_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
