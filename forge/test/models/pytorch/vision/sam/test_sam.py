# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.sam.utils.model import SamWrapper, get_model_inputs
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.xfail()
@pytest.mark.parametrize(
    "variant",
    [
        "facebook/sam-vit-huge",
        "facebook/sam-vit-large",
        "facebook/sam-vit-base",
    ],
)
@pytest.mark.nightly
def test_sam(forge_property_recorder, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="sam",
        variant=variant,
        task=Task.IMAGE_SEGMENTATION,
        source=Source.GITHUB,
    )

    # Record Forge Property

    if variant == "facebook/sam-vit-base":
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")

    forge_property_recorder.record_model_name(module_name)

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
