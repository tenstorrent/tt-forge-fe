# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.sam.utils.model import SamWrapper, get_model_inputs


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("facebook/sam-vit-huge", marks=pytest.mark.skip(reason="Skipping due to CI/CD Limitations")),
        pytest.param("facebook/sam-vit-large", marks=pytest.mark.skip(reason="Skipping due to CI/CD Limitations")),
        pytest.param("facebook/sam-vit-base", marks=pytest.mark.xfail()),
    ],
)
@pytest.mark.nightly
def test_sam(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="sam",
        variant=variant,
        task=Task.IMAGE_SEGMENTATION,
        source=Source.GITHUB,
    )

    if variant == "facebook/sam-vit-base":
        forge_property_recorder.record_group("red")
        forge_property_recorder.record_priority("P2")
    else:
        forge_property_recorder.record_group("generality")

    # Load  model and input

    framework_model, sample_inputs = get_model_inputs(variant)

    # Forge compile framework model
    framework_model = SamWrapper(framework_model)
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=sample_inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify(sample_inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
