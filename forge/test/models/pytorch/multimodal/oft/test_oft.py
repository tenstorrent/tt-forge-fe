# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.multimodal.oft.utils.oft_utils import (
    StableDiffusionWrapper,
    get_inputs,
)
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.xfail()
@pytest.mark.nightly
def test_oft(forge_property_recorder):
    # Build module name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="oft",
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_model_name(module_name)

    # Load model and inputs
    pipe, inputs = get_inputs()

    # Forge compile framework model
    wrapped_model = StableDiffusionWrapper(pipe)
    compiled_model = forge.compile(
        wrapped_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_recorder=forge_property_recorder,
    )

    # Model Verification
    verify(inputs, wrapped_model, compiled_model, forge_property_recorder=forge_property_recorder)
