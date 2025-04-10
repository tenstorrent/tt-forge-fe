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
from urllib.request import urlopen
from PIL import Image
from test.models.models_utils import print_cls_results
from forge.forge_property_utils import Framework, Source, Task

params = [
    pytest.param("efficientnet_b0"),
    pytest.param("efficientnet_b1", marks=[pytest.mark.push]),
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
def test_efficientnet_onnx(variant, forge_property_recorder, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="efficientnet",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    if variant == "efficientnet_b0":
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")

    # Load efficientnet model
    model = timm.create_model(variant, pretrained=True)

    # Load the inputs
    img = Image.open(
        urlopen("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png")
    )

    inputs = load_inputs(img, model)
    onnx_path = f"{tmp_path}/efficientnet.onnx"
    torch.onnx.export(model, inputs[0], onnx_path)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, forge_property_handler=forge_property_recorder)

    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
