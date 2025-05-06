# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from torchvision import transforms
import requests
from PIL import Image
import onnx

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task

variants = ["ddrnet23s", "ddrnet23", "ddrnet39"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_ddrnet(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="ddrnet",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    model_name = f"{variant}_onnx"

    # STEP 3: Prepare input
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

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

    # Load onnx model
    load_path = f"third_party/confidential_customer_models/generated/files/{variant}.onnx"
    model_name = f"{variant}_onnx"
    onnx_model = onnx.load(load_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


variants = ["ddrnet_23_slim_1024"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Dependent on CCM Repo")
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_ddrnet_semantic_segmentation_onnx(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="ddrnet",
        variant=variant,
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Prepare input
    image_path = "third_party/confidential_customer_models/cv_demos/ddrnet/semantic_segmentation/image/road_scenes.png"
    input_image = Image.open(image_path)
    input_image = transforms.Resize((1024, 1024))(input_image)
    input_tensor = transforms.ToTensor()(input_image)
    input_batch = input_tensor.unsqueeze(0)
    inputs = [input_batch]

    # Load and validate the model
    load_path = f"third_party/confidential_customer_models/customer/model_0/files/cnn/ddrnet/{variant}.onnx"
    model_name = f"{variant}_onnx"
    onnx_model = onnx.load(load_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(model_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
