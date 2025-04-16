# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import onnx

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.utils.yolovx_utils import (
    WorldModelWrapper,
    get_test_input,
)


@pytest.mark.push
@pytest.mark.xfail
@pytest.mark.nightly
def test_yolo_world_inference_onnx(forge_property_recorder, tmp_path):

    model_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt"

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="yolo_world",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Record Forge property

    forge_property_recorder.record_group("red")

    # Load framework_model and input

    torch_model = WorldModelWrapper(model_url)
    inputs = get_test_input()

    # Export model to ONNX
    onnx_path = tmp_path / "yolov10.onnx"
    torch.onnx.export(
        torch_model,
        inputs,
        onnx_path,
        input_names=["image"],
        output_names=["output"],
        opset_version=17,
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
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify([inputs], framework_model, compiled_model, forge_property_handler=forge_property_recorder)
