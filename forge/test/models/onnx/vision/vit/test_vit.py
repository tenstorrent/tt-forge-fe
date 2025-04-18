# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from datasets import load_dataset
import torch
import onnx
import forge
from transformers import ViTForImageClassification
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task


variants = [
    pytest.param("google/vit-base-patch16-224"),
    pytest.param(
        "google/vit-large-patch16-224",
        marks=[pytest.mark.skip(reason="Transient failure - Out of memory due to other tests in CI pipeline")],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vit_classify_224(forge_property_recorder, variant, tmp_path):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="vit_base",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    if variant in ["google/vit-base-patch16-224"]:
        forge_property_recorder.record_group("red")
        forge_property_recorder.record_priority("P1")
    else:
        forge_property_recorder.record_group("generality")

    # Load the torch model
    torch_model = ViTForImageClassification.from_pretrained(variant)

    # Load the inputs
    inputs = [torch.rand(1, 3, 224, 224)]

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
