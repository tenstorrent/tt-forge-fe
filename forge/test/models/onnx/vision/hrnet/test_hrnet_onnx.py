# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)

from test.models.models_utils import print_cls_results, preprocess_inputs
from test.utils import download_model
import onnx
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

variants = [
    "hrnet_w18_small_v1",
    "hrnet_w18_small_v2",
    "hrnetv2_w18",
    "hrnetv2_w30",
    "hrnetv2_w44",
    "hrnetv2_w48",
    "hrnetv2_w64",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_hrnet_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.HRNET,
        variant=variant,
        source=Source.OSMR,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model
    torch_model = download_model(ptcv_get_model, variant, pretrained=True)
    torch_model.eval()

    # Load input
    inputs = preprocess_inputs()

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant}.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    pcc = 0.99
    if variant == "hrnetv2_w64":
        pcc = 0.98

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    # Model Verification and Inference
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
