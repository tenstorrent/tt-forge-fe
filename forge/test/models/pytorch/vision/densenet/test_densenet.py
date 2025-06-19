# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torch.nn as nn
import torchxrayvision as xrv
from torchxrayvision.models import fix_resolution, op_norm

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
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.densenet.model_utils.densenet_utils import (
    get_input_img,
    get_input_img_hf_xray,
)
from test.utils import download_model

variants = [
    pytest.param(
        "densenet121",
    ),
    pytest.param(
        "densenet121_hf_xray",
        marks=[pytest.mark.xfail],
    ),
]


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
@pytest.mark.parametrize("variant", variants)
def test_densenet_121_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DENSENET,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # STEP 2: Create Forge module from PyTorch model
    if variant == "densenet121":
        framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet121", pretrained=True)
        img_tensor = get_input_img()
    else:
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        model_name = "densenet121-res224-all"
        model = download_model(xrv.models.get_model, model_name)
        framework_model = densenet_xray_wrapper(model)
        img_tensor = get_input_img_hf_xray()

    # STEP 3: Run inference on Tenstorrent device
    inputs = [img_tensor.to(torch.bfloat16)]
    framework_model.to(torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.97)),
    )

    # post processing
    if variant == "densenet121_hf_xray":
        outputs = op_norm(co_out[0], model.op_threshs)
    else:
        print_cls_results(fw_out[0], co_out[0])


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        "densenet161",
    ],
)
def test_densenet_161_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DENSENET,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet161", pretrained=True).to(
        torch.bfloat16
    )

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()
    inputs = [img_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Post Processing
    print_cls_results(fw_out[0], co_out[0])


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        "densenet169",
    ],
)
def test_densenet_169_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DENSENET,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet169", pretrained=True).to(
        torch.bfloat16
    )

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()

    inputs = [img_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Post Processing
    print_cls_results(fw_out[0], co_out[0])


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["densenet201"])
def test_densenet_201_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DENSENET,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "densenet201", pretrained=True).to(
        torch.bfloat16
    )

    # STEP 3: Run inference on Tenstorrent device
    img_tensor = get_input_img()

    inputs = [img_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    verify_cfg = VerifyConfig()
    if variant == "densenet201":
        verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))

    # Model Verification and Inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
