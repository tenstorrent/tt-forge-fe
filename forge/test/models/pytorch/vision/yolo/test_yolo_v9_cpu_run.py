# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from test.models.pytorch.vision.yolo.model_utils.yolo_utils import (
    YoloWrapper,
    load_yolo_model_and_image,
    postprocess
)


@pytest.mark.nightly
def test_yolov9():
   
    # Load  model and input
    model, image_tensor = load_yolo_model_and_image(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c.pt"
    )
    framework_model = YoloWrapper(model)
    
    co_out = framework_model(image_tensor)
    
    # Post process
    postprocess(co_out[0])
