# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import onnx
import pytest
from PIL import Image
from torchvision import transforms
import urllib

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task

variants = ["hardnet68", "hardnet85", "hardnet68ds", "hardnet39ds"]


@pytest.mark.skip_model_analysis
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.skip(reason="Dependent on CCM Repo")
def test_hardnet_onnx(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="hardnet",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Download an example image
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    # Preprocessing
    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    img_tensor = input_tensor.unsqueeze(0)
    inputs = [img_tensor]

    # Load ONNX model
    load_path = f"third_party/confidential_customer_models/generated/files/{variant}.onnx"
    model_name = f"hardnet_{variant}_onnx"
    onnx_model = onnx.load(load_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
