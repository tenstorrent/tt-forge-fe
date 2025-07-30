# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.fuyu.pytorch.loader import ModelLoader, ModelVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(ModelVariant.FUYU_8B),
    ],
)
def test_fuyu8b(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.FUYU,
        variant=variant.value,
        task=Task.QA,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs using model loader
    model_loader = ModelLoader(variant)
    framework_model = model_loader.load_model()
    inputs = model_loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=[inputs], module_name=module_name)

    # Model Verification
    verify([inputs], framework_model, compiled_model)
