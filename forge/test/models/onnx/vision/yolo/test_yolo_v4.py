# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx
import forge

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from third_party.tt_forge_models.yolov4 import ModelLoader  # isort:skip


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        x, y, z = self.model(image)
        # Post processing inside model casts output to float32,
        # even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return x.to(image.dtype), y.to(image.dtype), z.to(image.dtype)


@pytest.mark.nightly
@pytest.mark.xfail
def test_yolo_v4(forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.YOLOV4,
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Load model and input
    loader = ModelLoader()
    framework_model = loader.load_model()
    framework_model = Wrapper(framework_model)
    input_sample = loader.load_inputs()

    # Export to ONNX
    onnx_path = f"{forge_tmp_path}/yolov3.onnx"
    torch.onnx.export(
        framework_model,
        input_sample,
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
        sample_inputs=[input_sample],
        module_name=module_name,
    )

    # Model Verification
    verify(
        [input_sample],
        onnx_module,
        compiled_model,
    )
