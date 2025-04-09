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
from utils import load_inputs


@pytest.mark.nightly
@pytest.mark.xfail()
def test_unet_onnx(forge_property_recorder, tmp_path):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX, model="unet", variant="base", source=Source.TORCH_HUB, task=Task.IMAGE_SEGMENTATION
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    # TODO: this needs to be added
    # forge_property_recorder.record_priority("p1")
    forge_property_recorder.record_model_name(module_name)

    # Load the torch model
    torch_model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    torch_model.eval()

    # Load the inputs
    url, filename = (
        "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
        "TCGA_CS_4944.png",
    )
    inputs = load_inputs(url, filename)

    onnx_path = f"{tmp_path}/unet.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
