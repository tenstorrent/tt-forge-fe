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
from third_party.tt_forge_models.albert.token_classification.pytorch import (
    ModelLoader as TokenClassificationLoader,
)
from third_party.tt_forge_models.albert.token_classification.pytorch import (
    ModelVariant as TokenClassificationVariant,
)

token_classification_params = [
    pytest.param(TokenClassificationVariant.XLARGE_V2),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", token_classification_params)
def test_albert_token_classification_pytorch(variant):
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.albert.encoder.albert_layer_groups[0]
            self.head_mask = [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]

        def forward(self, hidden_states, attn_mask):

            output = self.model(hidden_states, attn_mask, self.head_mask)
            return output

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ALBERT,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load Model and inputs
    loader = TokenClassificationLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False

    framework_model = Wrapper(framework_model)

    logger.info("framework_model={}", framework_model)

    hidden_states = torch.load("hidden_states.pt")
    attn_mask = torch.load("attention_mask.pt")

    inputs = [hidden_states, attn_mask]

    logger.info("inputs={}", inputs)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )
