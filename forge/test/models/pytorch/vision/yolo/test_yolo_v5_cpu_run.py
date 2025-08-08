# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

from test.models.pytorch.vision.yolo.model_utils.yolo_utils import data_postprocessing, load_yolov5_model_and_image


size = [
    pytest.param("n", id="yolov5n"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size)
def test_yolov5_320x320(restore_package_versions, size):

    # prepare model and input
    framework_model, ims, n, files, shape0, shape1, pixel_values = load_yolov5_model_and_image(
        "ultralytics/yolov5",
        size=size,
        input_size=320
    )
    
    output = framework_model(pixel_values)
    
    # Data postprocessing on Host
    results = data_postprocessing(
        ims,
        pixel_values.shape,
        output,
        framework_model,
        n,
        shape0,
        shape1,
        files,
    )

    print("Predictions:\n", results.pandas().xyxy)
