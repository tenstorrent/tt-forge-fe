# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import RegNetForImageClassification, RegNetModel

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.regnet.utils.image_utils import preprocess_input_data
from test.models.pytorch.vision.utils.utils import load_vision_model_and_input
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["facebook/regnet-y-040"])
def test_regnet(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="regnet",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    # Load RegNet model
    framework_model = RegNetModel.from_pretrained("facebook/regnet-y-040")

    # Preprocess the image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = preprocess_input_data(image_url, variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["facebook/regnet-y-040"])
def test_regnet_img_classification(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="regnet",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    # Load the image processor and the RegNet model
    framework_model = RegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

    # Preprocess the image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = preprocess_input_data(image_url, variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants_with_weights = {
    "regnet_y_400mf": "RegNet_Y_400MF_Weights",
    "regnet_y_800mf": "RegNet_Y_800MF_Weights",
    "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights",
    "regnet_y_3_2gf": "RegNet_Y_3_2GF_Weights",
    "regnet_y_8gf": "RegNet_Y_8GF_Weights",
    "regnet_y_16gf": "RegNet_Y_16GF_Weights",
    "regnet_y_32gf": "RegNet_Y_32GF_Weights",
    "regnet_y_128gf": "RegNet_Y_128GF_Weights",
    "regnet_x_400mf": "RegNet_X_400MF_Weights",
    "regnet_x_800mf": "RegNet_X_800MF_Weights",
    "regnet_x_1_6gf": "RegNet_X_1_6GF_Weights",
    "regnet_x_3_2gf": "RegNet_X_3_2GF_Weights",
    "regnet_x_8gf": "RegNet_X_8GF_Weights",
    "regnet_x_16gf": "RegNet_X_16GF_Weights",
    "regnet_x_32gf": "RegNet_X_32GF_Weights",
}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_regnet_torchvision(record_forge_property, variant):

    if variant != "regnet_y_400mf":
        pytest.skip("Skipping this variant; only testing the small variant(regnet_y_400mf) for now.")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="regnet",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
