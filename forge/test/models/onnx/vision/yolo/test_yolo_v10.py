# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

import forge
from forge.verify.verify import verify

from test.models.onnx.vision.yolo.utils.yolo_utils import load_yolo_model_and_image, YoloWrapper
from forge.forge_property_utils import Framework, Source, Task


@pytest.mark.xfail()
@pytest.mark.nightly
def test_yolov10(forge_property_recorder, tmp_path):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="Yolov10",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P1")

    # Load  model and input
    model, image_tensor = load_yolo_model_and_image(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"
    )
    torch_model = YoloWrapper(model)

    # Export model to ONNX
    onnx_path = tmp_path / "yolov10.onnx"
    torch.onnx.export(
        torch_model,
        image_tensor,
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
        sample_inputs=[image_tensor],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify([image_tensor], framework_model, compiled_model, forge_property_handler=forge_property_recorder)
