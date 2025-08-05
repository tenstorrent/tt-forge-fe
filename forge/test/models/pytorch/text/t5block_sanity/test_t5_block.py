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
from forge.verify.verify import DeprecatedVerifyConfig, verify
from third_party.tt_forge_models.t5.pytorch import ModelLoader, ModelVariant


class T5Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, attention_mask, cache_position):
        output = self.model(hidden_states=hidden_states, attention_mask=attention_mask, cache_position=cache_position)
        return output


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
    hidden_states = torch.load("./hidden_states_t5block.pt")
    attention_mask = torch.load("./attention_mask_t5block.pt")
    cache_position = torch.load("./cache_position_t5block.pt")
    inputs = [hidden_states, attention_mask, cache_position]

    # Create wrapper for the model
    framework_model = model.decoder.block[0]
    framework_model = T5Wrapper(framework_model)
    # framework_model = model.decoder.block[:5]
    logger.info(f"Framework model: {framework_model}")

    # Forge compile
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        verify_cfg=DeprecatedVerifyConfig(verify_forge_codegen_vs_framework=True),
    )
    # Model Verification
    verify(inputs, framework_model, compiled_model)
