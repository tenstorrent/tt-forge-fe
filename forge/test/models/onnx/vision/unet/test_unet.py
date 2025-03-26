# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import numpy as np
import forge
import onnx
from forge.verify.verify import verify
from test.models.utils import Framework, Source, Task, build_module_name
from utils import load_inputs, load_model


@pytest.mark.xfail(
    reason="RuntimeError: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=1)] grow to 1691424 B which is beyond max L1 size of 1499136 B"
)
@pytest.mark.nightly
def test_unet_onnx(forge_property_recorder, tmp_path):

    # Build Module Name

    module_name = build_module_name(
        framework=Framework.ONNX, model="unet", source=Source.TORCH_HUB, task=Task.IMAGE_SEGMENTATION
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    # Load the torch model
    torch_model = load_model("unet")

    # Load the inputs
    url, filename = (
        "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
        "TCGA_CS_4944.png",
    )
    inputs = load_inputs(url, filename)

    onnx_path = f"{tmp_path}/unet.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
