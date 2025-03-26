# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from datasets import load_dataset
import torch
import onnx
import forge
from forge.verify.verify import verify
from utils import load_model, load_inputs
from test.models.utils import Framework, Source, Task, build_module_name


variants = [
    pytest.param(
        "google/vit-base-patch16-224",
        marks=[
            pytest.mark.xfail(
                reason="Out of Memory: Not enough space to allocate 12500992 B L1 buffer across 7 banks, where each bank needs to store 1785856 B"
            )
        ],
    ),
    pytest.param(
        "google/vit-large-patch16-224",
        marks=[
            pytest.mark.xfail(
                reason="Out of Memory: Not enough space to allocate 12500992 B L1 buffer across 7 banks, where each bank needs to store 1785856 B"
            )
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vit_classify_224(forge_property_recorder, variant, tmp_path):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="vit",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    if variant in ["google/vit-base-patch16-224"]:
        forge_property_recorder.record_group("priority")
    else:
        forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load the torch model
    torch_model = load_model(variant)
    url = "https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/elephant-299.jpg"
    filename = "elephant-299.jpg"

    # Load the inputs
    inputs = load_inputs(url, filename, variant)

    onnx_path = f"{tmp_path}/vit.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
