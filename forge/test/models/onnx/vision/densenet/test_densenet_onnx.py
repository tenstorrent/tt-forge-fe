# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.densenet.model_utils.densenet_utils import (
    get_input_img,
)
from test.utils import download_model
import onnx
import forge

variants = [pytest.param("densenet121", marks=pytest.mark.push), "densenet161", "densenet169"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_densenet_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DENSENET,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model
    torch_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", variant, pretrained=True)
    torch_model.eval()

    # Load and pre-process image
    img_tensor = get_input_img()
    inputs = [img_tensor]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
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
