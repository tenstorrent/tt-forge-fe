# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.beit.utils.utils import load_input, load_model

variants = [
    pytest.param(
        "microsoft/beit-base-patch16-224",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "microsoft/beit-large-patch16-224",
        marks=[pytest.mark.xfail],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_beit_image_classification(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="beit",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load model and input
    framework_model = load_model(variant)
    inputs = load_input(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
