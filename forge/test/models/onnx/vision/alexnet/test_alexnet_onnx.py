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
from test.models.models_utils import print_cls_results, preprocess_inputs
from test.utils import download_model


@pytest.mark.push
@pytest.mark.nightly
def test_alexnet_onnx(forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.ALEXNET,
        source=Source.TORCH_HUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model
    torch_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "alexnet", pretrained=True)
    torch_model.eval()

    # Load input
    inputs = preprocess_inputs()

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/alexnet.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    print_cls_results(fw_out[0], co_out[0])
