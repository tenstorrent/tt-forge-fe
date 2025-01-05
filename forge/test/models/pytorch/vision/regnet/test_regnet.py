# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoImageProcessor, RegNetModel, RegNetForImageClassification
import forge
from forge.verify.verify import verify
from .utils.image_utils import preprocess_input_data
from test.models.utils import build_module_name, Framework, Task


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", ["facebook/regnet-y-040"])
def test_regnet(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="regnet", variant=variant)

    record_forge_property("module_name", module_name)

    # Load RegNet model
    framework_model = RegNetModel.from_pretrained("facebook/regnet-y-040")

    # Preprocess the image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = preprocess_input_data(image_url, variant)

    # Compiler test
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", ["facebook/regnet-y-040"])
def test_regnet_img_classification(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="regnet", variant=variant, task=Task.IMAGE_CLASSIFICATION
    )

    record_forge_property("module_name", module_name)

    # Load the image processor and the RegNet model
    framework_model = RegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

    # Preprocess the image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = preprocess_input_data(image_url, variant)

    # Compiler test
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)
