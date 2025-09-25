# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.speecht5.pytorch import ModelLoader, ModelVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_values):
        return self.model(input_ids, attention_mask, decoder_input_values)[0]  # Return only the spectrogram tensor


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            ModelVariant.TTS,
        ),
    ],
)
def test_speecht5_tts(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SPEECHT5TTS,
        variant=variant,
        task=Task.TEXT_TO_SPEECH,
        source=Source.HUGGINGFACE,
    )

    # load_model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    model.config.return_dict = False
    model.config.use_cache = False
    framework_model = Wrapper(model)
    inputs_dict = loader.load_inputs()
    inputs = [inputs_dict["input_ids"], inputs_dict["attention_mask"], inputs_dict["decoder_input_values"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
