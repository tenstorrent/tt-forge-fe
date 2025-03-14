# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.beit.utils.utils import load_input, load_model
from test.models.utils import Framework, Source, Task, build_module_name

variants = [
    pytest.param(
        "microsoft/beit-base-patch16-224",
        marks=[
            pytest.mark.xfail(reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen")
        ],
    ),
    pytest.param(
        "microsoft/beit-large-patch16-224",
        marks=[
            pytest.mark.xfail(reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen")
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_beit_image_classification(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="beit",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load model and input
    framework_model = load_model(variant)
    inputs = load_input(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
