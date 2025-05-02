# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.utils.yolo_utils import (
    YoloWrapper,
    load_yolo_model_and_image,
)


@pytest.mark.xfail(
    reason="RuntimeError: Out of Memory: Not enough space to allocate 57843712 B L1 buffer across 64 banks, where each bank needs to store 903808 B"
)
@pytest.mark.nightly
def test_yolov8(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="Yolov8",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P2")

    # Load  model and input
    model, image_tensor = load_yolo_model_and_image(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    )
    framework_model = YoloWrapper(model)
    input = [image_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify(input, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
