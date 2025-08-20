# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.whisper.pytorch.loader import ModelLoader, ModelVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

variants = [ModelVariant.WHISPER_LARGE_V3]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail
def test_whisper_large_v3(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.WHISPER,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    input_features = loader.load_inputs()

    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    inputs = [input_features, decoder_input_ids]

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features, decoder_input_ids):
            inputs = {"input_features": input_features, "decoder_input_ids": decoder_input_ids}
            output = self.model(**inputs)
            return output

    framework_model = Wrapper(model)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
