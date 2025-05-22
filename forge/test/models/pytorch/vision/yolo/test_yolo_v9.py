# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.model_utils.yolo_utils import (
    YoloWrapper,
    load_yolo_model_and_image,
)


@pytest.mark.nightly
def test_yolov9():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="Yolov9",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Load  model and input
    model, image_tensor = load_yolo_model_and_image(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c.pt"
    )
    framework_model = YoloWrapper(model).to(torch.bfloat16)
    input = [image_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b

    compiler_cfg = CompilerConfig(default_df_override=data_format_override)
    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(input, framework_model, compiled_model)
