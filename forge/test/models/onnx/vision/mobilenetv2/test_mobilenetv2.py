# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import timm
import forge
import onnx
import torch
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig, AutomaticValueChecker
from test.models.models_utils import print_cls_results, load_timm_model_and_input
from forge.forge_property_utils import Framework, Source, Task

params = [
    pytest.param("mobilenetv2_050"),
    pytest.param("mobilenetv2_100", marks=[pytest.mark.push]),
    pytest.param("mobilenetv2_110d"),
    pytest.param("mobilenetv2_140"),
]


@pytest.mark.parametrize("variant", params)
@pytest.mark.nightly
def test_mobilenetv2_onnx(variant, forge_property_recorder, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="mobilenetv2",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    if variant == "mobilenetv2_050":
        forge_property_recorder.record_group("generality")
        forge_property_recorder.record_priority("P1")
    else:
        forge_property_recorder.record_group("generality")

    # Load the model and inputs
    model, inputs = load_timm_model_and_input(variant)
    onnx_path = f"{tmp_path}/mobilenetv2.onnx"
    torch.onnx.export(model, inputs[0], onnx_path)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(
        onnx_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    pcc = 0.99
    if variant == "mobilenetv2_050":
        pcc = 0.96

    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )

    # # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
