# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import requests
from yolov6 import YOLOV6

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.utils.yolov6_utils import (
    check_img_size,
    process_image,
)

# Didn't dealt with yolov6n6,yolov6s6,yolov6m6,yolov6l6 variants because of its higher input size(1280)
variants = [
    "yolov6n",
    "yolov6s",
    "yolov6m",
    "yolov6l",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolo_v6_pytorch(forge_property_recorder, variant):
    if variant != "yolov6n":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="yolo_v6",
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
    )

    # STEP 2 :prepare model
    url = f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{variant}.pt"
    weights = f"{variant}.pt"

    try:
        response = requests.get(url)
        with open(weights, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {url} to {weights}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

    model = YOLOV6(weights)
    framework_model = model.model
    framework_model.eval()

    # STEP 3 : prepare input
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    stride = 32
    input_size = 640
    img_size = check_img_size(input_size, s=stride)
    img, img_src = process_image(url, img_size, stride, half=False)
    input_batch = img.unsqueeze(0)

    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # STEP 5 : remove downloaded weights
    os.remove(weights)
