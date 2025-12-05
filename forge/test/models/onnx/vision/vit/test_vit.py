# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from datasets import load_dataset
import torch
import onnx
import forge
from transformers import ViTForImageClassification, AutoImageProcessor
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties


variants = [
    pytest.param("google/vit-base-patch16-224", marks=pytest.mark.pr_models_regression),
    pytest.param(
        "google/vit-large-patch16-224",
        marks=[
            pytest.mark.out_of_memory,
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vit_classify_224(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.VITBASE,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    if variant == "google/vit-large-patch16-224":
        pytest.xfail(reason="Requires multi-chip support")

    # Load torch model and processor
    torch_model = ViTForImageClassification.from_pretrained(variant)
    image_processor = AutoImageProcessor.from_pretrained(variant)

    # prepare input
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    inputs = [image_processor(image, return_tensors="pt").pixel_values]

    onnx_path = f"{forge_tmp_path}/vit.onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    pcc = 0.99
    if variant == "google/vit-base-patch16-224":
        pcc = 0.95

    # Model Verification and Inference
    _, co_out = verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    # post processing
    logits = co_out[0]
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", torch_model.config.id2label[predicted_class_idx])
