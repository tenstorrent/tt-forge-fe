# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import timm
import forge
import onnx
import torch
from forge.verify.verify import verify
from datasets import load_dataset
from forge.verify.config import VerifyConfig, AutomaticValueChecker
from utils import load_inputs
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import print_cls_results

variants = [
    "mobilenetv2_050",
    "mobilenetv2_100",
    "mobilenetv2_110d",
    "mobilenetv2_140",
]


@pytest.mark.push
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_mobilenetv2_onnx(variant, forge_property_recorder, tmp_path):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="mobilenetv2",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    if variants == "mobilenetv2_050":
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load the inputs
    dataset = load_dataset("huggingface/cats-image")
    img = dataset["test"]["image"][0]
    inputs = load_inputs(img)

    # Load mobilenetv2 model
    model = timm.create_model(variant, pretrained=True)
    onnx_path = f"{tmp_path}/mobilenetv2.onnx"
    torch.onnx.export(model, inputs[0], onnx_path, opset_version=17)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, forge_property_handler=forge_property_recorder)

    pcc = 0.99
    if variant == "mobilenetv2_050":
        pcc = 0.98

    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )

    # # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
