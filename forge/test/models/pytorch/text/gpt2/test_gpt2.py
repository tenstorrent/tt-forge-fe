# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.gpt2.pytorch import ModelLoader

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        "gpt2",
    ],
)
def test_gpt2_text_gen(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GPT,
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Load Inputs and Model
    loader = ModelLoader()
    model = loader.load_model()
    inputs = loader.load_inputs()
    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, model, compiled_model)
