# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.utils.yolo_utils import (
    YoloWrapper,
    load_yolo_model_and_image,
)


@pytest.mark.xfail
@pytest.mark.nightly
def test_yolov10(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="Yolov10",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P1")

    # Load  model and input
    model, image_tensor = load_yolo_model_and_image(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"
    )
    framework_model = YoloWrapper(model)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[image_tensor],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify([image_tensor], framework_model, compiled_model, forge_property_handler=forge_property_recorder)
