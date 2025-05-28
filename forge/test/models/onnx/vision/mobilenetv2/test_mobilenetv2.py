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
from test.models.onnx.vision.mobilenetv2.model_utils.utils import load_inputs
from urllib.request import urlopen
from PIL import Image
from test.models.models_utils import print_cls_results
from forge.forge_property_utils import Framework, Source, Task, ModelPriority, ModelArch, record_model_properties

params = [
    pytest.param("mobilenetv2_050"),
    pytest.param("mobilenetv2_100", marks=[pytest.mark.push]),
    pytest.param("mobilenetv2_110d"),
    pytest.param("mobilenetv2_140"),
]


@pytest.mark.parametrize("variant", params)
@pytest.mark.nightly
def test_mobilenetv2_onnx(variant, forge_tmp_path):

    priority = ModelPriority.P1 if variant == "mobilenetv2_050" else ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.MOBILENETV2,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
        priority=priority,
    )

    # Load mobilenetv2 model
    model = timm.create_model(variant, pretrained=True)

    # Load the inputs
    img = Image.open(
        urlopen("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png")
    )

    inputs = load_inputs(img, model)
    onnx_path = f"{forge_tmp_path}/mobilenetv2.onnx"
    torch.onnx.export(model, inputs[0], onnx_path)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    pcc = 0.99
    if variant == "mobilenetv2_050":
        pcc = 0.96

    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
