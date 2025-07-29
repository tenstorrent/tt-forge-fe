# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import torch.nn as nn
import torchxrayvision as xrv
from third_party.tt_forge_models.densenet.pytorch import ModelLoader, ModelVariant
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

from test.models.pytorch.vision.densenet.model_utils.densenet_utils import (
    get_input_img_hf_xray,
)
from test.utils import download_model

variants = [
    pytest.param(
        "densenet121_hf_xray",
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
def test_densenet_121_hf_xray_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DENSENET,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # STEP 2: Create Forge module from PyTorch model
    op_threshs = None
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    model_name = "densenet121-res224-all"
    model = download_model(xrv.models.get_model, model_name)
    framework_model = densenet_xray_wrapper(model)
    img_tensor = get_input_img_hf_xray()
    op_threshs = model.op_threshs

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
    op_norm(co_out[0].to(torch.float32), op_threshs.to(torch.float32))


variants = [
    ModelVariant.DENSENET121,
    ModelVariant.DENSENET161,
    ModelVariant.DENSENET169,
    ModelVariant.DENSENET201,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_densenet_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DENSENET,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    pcc = 0.99
    if variant == ModelVariant.DENSENET121:
        pcc = 0.97
    elif variant == ModelVariant.DENSENET201:
        pcc = 0.95

    # Model Verification and Inference
    _, co_out = verify(
        inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    # Post Processing
    loader.print_cls_results(co_out)
