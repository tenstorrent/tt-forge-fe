# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from ultralytics import YOLO

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.model_utils.yolovx_utils import get_test_input


class YoloWorldWrapper(torch.nn.Module):
    def __init__(self, model_url: str):
        super().__init__()
        self.yolo = YOLO(model_url)

    def forward(self, x):
        return self.yolo.model.forward(x, augment=False)


@pytest.mark.xfail
@pytest.mark.nightly
def test_yolo_world_inference():

    model_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt"

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="yolo_world",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
        group=ModelGroup.RED,
    )

    # Load framework_model and input
    framework_model = YoloWorldWrapper(model_url).to(torch.bfloat16)
    inputs = [get_test_input().to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile with Forge
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
