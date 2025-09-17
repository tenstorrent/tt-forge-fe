# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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

from test.models.pytorch.vision.yolo.model_utils.yolo_utils import (
    YoloWrapper,
    load_yolo_model_and_image,
)


@pytest.mark.nightly
def test_yolov9(forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.YOLOV9,
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Load  model and input
    model, image_tensor = load_yolo_model_and_image(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c.pt"
    )
    framework_model = YoloWrapper(model)
    inputs = [image_tensor]

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/yolov9.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_module = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(
        inputs,
        onnx_module,
        compiled_model,
    )
