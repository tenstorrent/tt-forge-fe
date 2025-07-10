# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

from test.utils import fetch_model, yolov5_loader

base_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0"


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, i1, i2, i3):
        ip = [i1, i2, i3]
        return self.model(ip)


def generate_model_yoloV5I320_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size

    model = fetch_model(name, f"{base_url}/{name}.pt", yolov5_loader, variant=variant)

    input_shape = (1, 3, 320, 320)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = [
    pytest.param("n", id="yolov5n"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size)
def test_yolov5_s(restore_package_versions, size):

    pcc = 0.99
    if size == "l":
        pcc = 0.95

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV5,
        variant="yolov5" + size,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="320x320",
    )

    framework_model, _, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    framework_model = framework_model.model.model.model[24]
    framework_model = Wrapper(framework_model)
    framework_model.to(torch.bfloat16)

    inputs = [
        torch.load("./debug_yolov5_inputs/input_tensor_0.pt").to(torch.bfloat16),
        torch.load("./debug_yolov5_inputs/input_tensor_1.pt").to(torch.bfloat16),
        torch.load("./debug_yolov5_inputs/input_tensor_2.pt").to(torch.bfloat16),
    ]

    logger.info("framework_model={}", framework_model)
    logger.info("inputs={}", inputs)

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
