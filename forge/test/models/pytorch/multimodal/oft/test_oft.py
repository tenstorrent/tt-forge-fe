# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from third_party.tt_forge_models.oft_stable_diffusion.pytorch import (
    ModelLoader,
    ModelVariant,
)

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
)


@pytest.mark.xfail
@pytest.mark.parametrize("variant", [ModelVariant.OFT_STABLE_DIFFUSION_V1_5])
@pytest.mark.nightly
def test_oft(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.OFT,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    pipe = loader.load_model()
    inputs = loader.load_inputs()

    # Forge compile framework model
    framework_model = StableDiffusionWrapper(pipe)
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
