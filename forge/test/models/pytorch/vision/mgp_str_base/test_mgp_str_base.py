# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# From: https://huggingface.co/alibaba-damo/mgp-str-base
import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.mgp_str_base.utils.utils import load_input, load_model


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("alibaba-damo/mgp-str-base", marks=[pytest.mark.xfail]),
    ],
)
def test_mgp_scene_text_recognition(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mgp",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.SCENE_TEXT_RECOGNITION,
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
