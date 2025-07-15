# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.multimodal.oft.model_utils.oft_utils import (
    StableDiffusionWrapper,
    get_inputs,
)


@pytest.mark.out_of_memory
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["runwayml/stable-diffusion-v1-5"])
@pytest.mark.nightly
def test_oft(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.OFT,
        variant=variant.split("/")[-1],
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and inputs
    pipe, inputs = get_inputs(model=variant)

    # Forge compile framework model
    framework_model = StableDiffusionWrapper(pipe)
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
