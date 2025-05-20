# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
from forge.forge_property_utils import Framework, ModelGroup, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.utils.yolovx_utils import (
    WorldModelWrapper,
    get_test_input,
)


@pytest.mark.xfail
@pytest.mark.nightly
def test_yolo_world_inference(forge_property_recorder):

    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt"

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="yolo_world",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
        group=ModelGroup.RED,
    )


    # Load framework_model and input

    framework_model = WorldModelWrapper(MODEL_URL)

    inputs = [get_test_input()]

    # Compile with Forge
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
