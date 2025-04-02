# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.utils.yolo_utils import (
    YoloWrapper,
    load_yolo_model_and_image,
)
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.xfail(reason="AssertionError: Encountered unsupported op types. Check error logs for more details")
@pytest.mark.nightly
def test_yolov10(forge_property_recorder):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="Yolov10",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_model_name(module_name)

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
