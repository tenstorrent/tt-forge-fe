# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

import forge
from forge.verify.verify import verify

from test.models.onnx.vision.yolo.utils.yolo10_utils import load_yolov10_model_and_image, YoloV10Wrapper
from test.models.utils import Framework, Source, Task, build_module_name

# Opset 10 is the minimum version to support DetectionModel in Torch.
# Opset 17 is the maximum version in Torchscript.
opset_versions = [10, 17]


@pytest.mark.parametrize("opset_version", opset_versions, ids=[str(v) for v in opset_versions])
@pytest.mark.xfail(reason="AssertionError: Encountered unsupported op types. Check error logs for more details")
@pytest.mark.nightly
def test_yolov10(forge_property_recorder, tmp_path, opset_version):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="Yolov10",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    # Load  model and input
    model, image_tensor = load_yolov10_model_and_image()
    torch_model = YoloV10Wrapper(model)

    # Export model to ONNX
    onnx_path = tmp_path / "yolov10.onnx"
    torch.onnx.export(
        torch_model,
        image_tensor,
        onnx_path,
        input_names=["image"],
        output_names=["output"],
        opset_version=opset_version,
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
