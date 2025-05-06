# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import onnx

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.model_utils.yolovx_utils import (
    WorldModelWrapper,
    get_test_input,
)
from forge.forge_property_utils import Framework, Source, Task, ModelPriority, ModelArch, record_model_properties


@pytest.mark.xfail
@pytest.mark.nightly
def test_yolo_world_inference_onnx(tmp_path):

    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt"

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.YOLOWORLD,
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
        priority=ModelPriority.P1,
    )

    # Load framework_model and input

    torch_model = WorldModelWrapper(MODEL_URL)
    inputs = get_test_input()

    # Export model to ONNX
    onnx_path = tmp_path / "yolo-world.onnx"
    torch.onnx.export(
        torch_model,
        inputs,
        onnx_path,
        input_names=["image"],
        output_names=["output"],
        dynamic_axes={"image": {0: "batch_size"}},
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=[inputs],
        module_name=module_name,
    )

    # Model Verification
    verify([inputs], framework_model, compiled_model)
