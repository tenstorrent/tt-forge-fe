# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

from test.models.pytorch.vision.yolo.model_utils.yolo_utils import (
    YoloWrapper,
    load_yolo_model_and_image,
    postprocess
)

variants = ["yolov8x", "yolov8n"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolov8(variant):

  
    # Load  model and input
    model, image_tensor = load_yolo_model_and_image(
        f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{variant}.pt"
    )
    framework_model = YoloWrapper(model)
    
    co_out = framework_model(image_tensor)
    
    # Post process
    postprocess(co_out[0])
