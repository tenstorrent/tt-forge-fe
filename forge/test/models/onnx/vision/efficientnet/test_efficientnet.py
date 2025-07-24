# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import timm
import forge
import onnx
import torch
from datasets import load_dataset
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig, AutomaticValueChecker
from test.models.onnx.vision.vision_utils import load_inputs
from test.models.models_utils import print_cls_results
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

params = [
    pytest.param("efficientnet_b0", marks=[pytest.mark.push]),
    pytest.param("efficientnet_b1"),
    pytest.param("efficientnet_b2"),
    pytest.param("efficientnet_b2a"),
    pytest.param("efficientnet_b3"),
    pytest.param("efficientnet_b3a"),
    pytest.param("efficientnet_b4"),
    pytest.param("efficientnet_b5"),
    pytest.param("efficientnet_lite0"),
]


@pytest.mark.parametrize("variant", params)
@pytest.mark.nightly
def test_efficientnet_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.EFFICIENTNET,
        variant=variant,
        source=Source.TIMM,
        task=Task.CV_IMAGE_CLS,
    )
    if variant == "efficientnet_b5":
        pytest.xfail(reason="Requires multi-chip support")

    # Load efficientnet model
    model = timm.create_model(variant, pretrained=True)

    # Load the inputs
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    img = next(iter(dataset.skip(10)))["image"]
    inputs = load_inputs(img, model)
    onnx_path = f"{forge_tmp_path}/efficientnet.onnx"
    torch.onnx.export(model, inputs[0], onnx_path)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    pcc = 0.99

    if variant == "efficientnet_b1":
        pcc = 0.95

    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(
            value_checker=AutomaticValueChecker(pcc=pcc),
        ),
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
