# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import requests
import onnx
from transformers import RegNetForImageClassification

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.regnet.model_utils.image_utils import (
    preprocess_input_data,
)


variants = [
    "regnety_040",
    "regnety_064",
    "regnety_080",
    "regnety_120",
    "regnety_160",
    "regnety_320",
]

hf_variant_map = {
    "regnety_040": "facebook/regnet-y-040",
    "regnety_064": "facebook/regnet-y-064",
    "regnety_080": "facebook/regnet-y-080",
    "regnety_120": "facebook/regnet-y-120",
    "regnety_160": "facebook/regnet-y-160",
    "regnety_320": "facebook/regnet-y-320",
}


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_regnet_img_classification(variant, forge_tmp_path):

    hf_variant = hf_variant_map[variant]

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.REGNET,
        variant=hf_variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    framework_model = RegNetForImageClassification.from_pretrained(hf_variant, return_dict=False)

    # Preprocess the image
    inputs = preprocess_input_data(hf_variant)
    inputs = [inputs[0]]

    opset_version = "17"
    onnx_path = os.path.join(forge_tmp_path, f"{variant}_Opset{opset_version}.onnx")

    if not os.path.exists(onnx_path):
        if not os.path.exists("regnet"):
            os.mkdir("regnet")
        url = f"https://github.com/onnx/models/raw/main/Computer_Vision/{variant}_Opset{opset_version}_timm/{variant}_Opset{opset_version}.onnx"
        response = requests.get(url, stream=True)
        response.raise_for_status()  # better to raise if download fails
        with open(onnx_path, "wb") as f:
            f.write(response.content)

    # Load Regnet model from ONNX
    model_name = f"regnet_{variant}_onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    logits = co_out[0]
    predicted_label = logits.argmax(-1).item()
    print(framework_model.config.id2label[predicted_label])
