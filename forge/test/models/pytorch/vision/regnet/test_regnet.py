# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import RegNetForImageClassification, RegNetModel

import forge
from forge.verify.verify import verify

from .utils.image_utils import preprocess_input_data
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
