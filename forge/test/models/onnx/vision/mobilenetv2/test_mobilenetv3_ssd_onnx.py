# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.mobilenet.model_utils.mobilenet_v3_ssd_utils import (
    load_input,
    load_model,
)
import onnx
import torch

variants_with_weights = {
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_mobilenetv3_ssd_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MOBILENETV3SSD,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.CV_IMAGE_CLS,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    torch_model = load_model(variant, weight_name)
    inputs = load_input()

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
