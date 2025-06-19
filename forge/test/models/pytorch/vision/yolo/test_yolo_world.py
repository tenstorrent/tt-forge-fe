# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.model_utils.yolovx_utils import (
    WorldModelWrapper,
    get_test_input,
    load_world_model,
)


@pytest.mark.xfail
@pytest.mark.nightly
def test_yolo_world_inference():

    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt"

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOWORLD,
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load framework_model and input
    framework_model = load_world_model(MODEL_URL)
    framework_model = WorldModelWrapper(framework_model).to(torch.bfloat16)
    inputs = [get_test_input().to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile with Forge
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
