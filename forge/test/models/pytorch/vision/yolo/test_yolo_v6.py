# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import pytest
import requests
import torch
from yolov6.layers.common import DetectBackend

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
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.model_utils.yolov6_utils import (
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


class YoloV6Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y, _ = self.model(x)
        # The model outputs float32, even if the input is bfloat16
        # Cast the output back to the input dtype
        return y.to(x.dtype)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolo_v6_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV6,
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

    model = DetectBackend(weights)
    framework_model = model.model
    framework_model.to(torch.bfloat16)
    framework_model = YoloV6Wrapper(framework_model)

    # STEP 3 : prepare input
    stride = 32
    input_size = 640
    img_size = check_img_size(input_size, s=stride)
    img, img_src = process_image(img_size, stride, half=False)
    input_batch = img.unsqueeze(0)

    inputs = [input_batch.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override, enable_optimization_passes=True)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    # STEP 5 : remove downloaded weights
    os.remove(weights)
