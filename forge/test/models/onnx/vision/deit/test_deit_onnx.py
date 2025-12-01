# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor, ViTForImageClassification

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model
import onnx
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

variants = [
    pytest.param("facebook/deit-base-patch16-224", marks=pytest.mark.push),
    "facebook/deit-small-patch16-224",
    "facebook/deit-tiny-patch16-224",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_deit_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DEIT,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model
    image_processor = download_model(AutoFeatureExtractor.from_pretrained, variant)
    torch_model = download_model(ViTForImageClassification.from_pretrained, variant, return_dict=False)
    torch_model.eval()

    # Prepare input
    dataset = load_dataset("huggingface/cats-image")
    image_1 = dataset["test"]["image"][0]
    img_tensor = image_processor(image_1, return_tensors="pt").pixel_values
    inputs = [img_tensor]

    # Export model to ONNX
    onnx_path = f'{forge_tmp_path}/{variant.split("/")[-1].replace("-", "_")}.onnx'
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, module_name=module_name)

    pcc = 0.99
    if variant == "facebook/deit-base-patch16-224":
        pcc = 0.96

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post processing
    logits = co_out[0]
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class: ", torch_model.config.id2label[predicted_class_idx])
