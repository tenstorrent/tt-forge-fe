# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from datasets import load_dataset
import forge
import onnx
import torch
from forge.verify.verify import verify
import shutil
from utils import load_inputs
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import print_cls_results
import timm

variants = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b2a",
    "efficientnet_b3",
    "efficientnet_b3a",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_lite0",
]


@pytest.mark.push
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_efficientnet_onnx(variant, forge_property_recorder, tmp_path):

    # Build Module Name
    module_name = build_module_name(
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

    forge_property_recorder.record_model_name(module_name)

    # Load the inputs
    dataset = load_dataset("huggingface/cats-image")
    img = dataset["test"]["image"][0]
    inputs = load_inputs(img)

    # Load efficientnet model
    model = timm.create_model(variant, pretrained=True)
    onnx_path = f"{tmp_path}/efficientnet.onnx"
    torch.onnx.export(model, inputs[0], onnx_path, opset_version=17)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify data on sample input
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
