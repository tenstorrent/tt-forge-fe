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

import os

import cv2
import pytest
import requests
import torch
from yolox.exp import get_exp

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

from test.models.pytorch.vision.yolo.model_utils.yolox_utils import preprocess

variants = [
    pytest.param("yolox_nano"),
    pytest.param("yolox_tiny"),
    pytest.param("yolox_s"),
    pytest.param("yolox_m"),
    pytest.param("yolox_l"),
    pytest.param("yolox_darknet"),
    pytest.param("yolox_x"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolox_pytorch(variant):

    pcc = 0.99
    if variant in ["yolox_nano", "yolox_l", "yolox_x"]:
        pcc = 0.97

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOX,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
    )

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
    framework_model = exp.get_model()
    ckpt = torch.load(f"{variant}.pth", map_location="cpu")
    framework_model.load_state_dict(ckpt["model"])

    # Set to false as it is part of model post-processing
    # to avoid pcc mismatch due to inplace slice and update
    framework_model.head.decode_in_inference = False

    framework_model.eval()
    framework_model.to(torch.bfloat16)
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

    inputs = [img_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # remove downloaded weights,image
    os.remove(weight_name)
    os.remove("input.jpg")
