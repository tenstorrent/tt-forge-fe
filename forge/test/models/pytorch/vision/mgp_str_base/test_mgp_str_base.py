# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# From: https://huggingface.co/alibaba-damo/mgp-str-base
import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.mgp_str_base.utils.utils import load_input, load_model
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "alibaba-damo/mgp-str-base", marks=[pytest.mark.xfail(reason="RuntimeError: Couldn't lower all tuples")]
        ),
    ],
)
def test_mgp_scene_text_recognition(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mgp",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.SCENE_TEXT_RECOGNITION,
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
