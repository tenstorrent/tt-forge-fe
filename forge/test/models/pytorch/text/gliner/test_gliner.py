# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from third_party.tt_forge_models.gliner.pytorch import ModelLoader, ModelVariant

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

from test.models.pytorch.text.gliner.model_utils.model_utils import GlinerWrapper

variants = [ModelVariant.GLINER_MULTI_V21]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_gliner(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GLINER,
        variant=variant.value,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.GITHUB,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model = GlinerWrapper(framework_model)
    inputs = loader.load_inputs()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    _, co_out = verify(inputs, framework_model, compiled_model)

    loader.post_processing(co_out)
