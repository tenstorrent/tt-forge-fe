# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.utils.yolovx_utils import (
    get_test_input,
    load_yolo_model,
)
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
def test_yolov8(record_forge_property):

    # Upgrade ultralytics version
    # install_ultralytics("8.3.91")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model=f"yolov8n",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    record_forge_property("group", "priority_2")
    record_forge_property("tags.model_name", module_name)

    # Load model and input
    framework_model = load_yolo_model("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt")
    image_tensor = get_test_input()

    # Forge compile framework model
    inputs = [image_tensor]
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
