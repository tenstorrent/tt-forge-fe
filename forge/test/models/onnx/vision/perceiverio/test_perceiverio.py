# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import onnx

import os
import requests
from PIL import Image
import pytest

from transformers import AutoImageProcessor

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.parametrize(
    "model_name",
    [
        "deepmind/vision-perceiver-conv",
        "deepmind/vision-perceiver-learned",
        "deepmind/vision-perceiver-fourier",
    ],
)
@pytest.mark.nightly
def test_perceiver_for_image_classification_onnx(forge_property_recorder, model_name):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="perceiverio",
        variant=model_name,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Sample Image
    pixel_values = get_sample_data(model_name)
    inputs = [pixel_values]

    # Load onnx model
    onnx_model_path = (
        "third_party/confidential_customer_models/generated/files/"
        + str(model_name).split("/")[-1].replace("-", "_")
        + ".onnx"
    )
    model_name = f"perceiver{model_name}_onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
