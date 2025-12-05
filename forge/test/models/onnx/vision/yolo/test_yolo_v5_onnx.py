# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import fetch_model, yolov5_loader

base_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0"


def generate_model_yoloV5I320_imgcls_torchhub_pytorch(variant, size):
    name = "yolov5" + size

    model = fetch_model(name, f"{base_url}/{name}.pt", yolov5_loader, variant=variant)

    input_shape = (1, 3, 320, 320)
    input_tensor = torch.rand(input_shape)
    return model, [input_tensor], {}


size = [
    pytest.param("n", id="yolov5n", marks=pytest.mark.pr_models_regression),
    pytest.param("s", id="yolov5s"),
    pytest.param("m", id="yolov5m"),
    pytest.param("l", id="yolov5l"),
    pytest.param(
        "x",
        id="yolov5x",
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size", size)
def test_yolov5_320x320(size, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.YOLOV5,
        variant="yolov5" + size,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="320x320",
    )

    framework_model, inputs, _ = generate_model_yoloV5I320_imgcls_torchhub_pytorch(
        "ultralytics/yolov5",
        size=size,
    )

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/yolov5{size}_320x320.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_module = forge.OnnxModule(module_name, onnx_model)

    # Compile ONNX
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Verify
    verify(
        inputs,
        onnx_module,
        compiled_model,
    )
