# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from third_party.tt_forge_models.t5.pytorch import ModelLoader, ModelVariant

variants = [
    pytest.param(ModelVariant.FLAN_T5_SMALL),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_t5_generation(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.T5,
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Skip multi-chip variants for now
    if variant not in [ModelVariant.SMALL, ModelVariant.FLAN_T5_SMALL, ModelVariant.BASE, ModelVariant.LARGE]:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()

    # Extract the inputs and prepare them for the model
    hidden_states = torch.load("./hidden_states_t5attention.pt")
    mask = torch.load("./mask_t5attention.pt")
    key_value_states = torch.load("./key_value_states_t5attention.pt")
    inputs = [hidden_states, mask, key_value_states]

    # Create wrapper for the model
    framework_model = model.decoder.block[0].layer[1].EncDecAttention
    logger.info(f"Framework model: {framework_model}")
    # Forge compile
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
