# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import subprocess

subprocess.run(
    ["pip", "install", "yolox==0.3.0", "--no-deps"]
)  # Install yolox==0.3.0 without installing its dependencies

"""
Reason to install yolox=0.3.0 through subprocess :
requirements of yolox=0.3.0 can be found here https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/requirements.txt
onnx==1.8.1 and onnxruntime==1.8.0 are required by yolox which are incompatible with our package versions
Dependencies required by yolox for pytorch implemetation are already present in pybuda and packages related to onnx is not needed
pip install yolox==0.3.0 --no-deps can be used to install a package without installing its dependencies through terminal
But in pybuda packages were installed through requirements.txt file not though terminal.
unfortunately there is no way to include --no-deps in  requirements.txt file.
for this reason , yolox==0.3.0 is intalled through subprocess.
"""

import torch
import cv2
import numpy as np
from yolox.exp import get_exp
import requests
import pytest
import os
import forge
from test.models.pytorch.vision.yolo.utils.yolox_utils import preprocess
from test.models.utils import build_module_name


variants = ["yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l", "yolox_darknet", "yolox_x"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_yolox_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="yolox", variant=variant)

    record_forge_property("module_name", module_name)

    # prepare model
    weight_name = f"{variant}.pth"
    url = f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{weight_name}"
    response = requests.get(url)
    with open(f"{weight_name}", "wb") as file:
        file.write(response.content)

    if variant == "yolox_darknet":
        model_name = "yolov3"
    else:
        model_name = variant.replace("_", "-")

    exp = get_exp(exp_name=model_name)
    model = exp.get_model()
    ckpt = torch.load(f"{variant}.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # Set to false as it is part of model post-processing
    # to avoid pcc mismatch due to inplace slice and update
    model.head.decode_in_inference = False

    model.eval()
    model_name = f"pt_{variant}"

    # prepare input
    if variant in ["yolox_nano", "yolox_tiny"]:
        input_shape = (416, 416)
    else:
        input_shape = (640, 640)

    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    response = requests.get(url)
    with open("input.jpg", "wb") as f:
        f.write(response.content)
    img = cv2.imread("input.jpg")
    img_tensor = preprocess(img, input_shape)
    img_tensor = img_tensor.unsqueeze(0)

    inputs = [img_tensor]

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    # remove downloaded weights,image
    os.remove(weight_name)
    os.remove("input.jpg")
