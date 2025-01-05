# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import os
import pytest
import requests
from yolov6 import YOLOV6
from test.models.pytorch.vision.yolo.utils.yolov6_utils import check_img_size, process_image
from test.models.utils import build_module_name


# Didn't dealt with yolov6n6,yolov6s6,yolov6m6,yolov6l6 variants because of its higher input size(1280)
variants = ["yolov6n", "yolov6s", "yolov6m", "yolov6l"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_yolo_v6_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="yolo_v6", variant=variant)

    record_forge_property("module_name", module_name)

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
    model = model.model
    model.eval()

    # STEP 3 : prepare input
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    stride = 32
    input_size = 640
    img_size = check_img_size(input_size, s=stride)
    img, img_src = process_image(url, img_size, stride, half=False)
    input_batch = img.unsqueeze(0)

    # STEP 4 : Inference
    compiled_model = forge.compile(model, sample_inputs=[input_batch], module_name=module_name)

    # STEP 5 : remove downloaded weights
    os.remove(weights)
