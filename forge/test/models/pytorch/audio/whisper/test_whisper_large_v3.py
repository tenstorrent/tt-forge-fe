# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Large V3 turbo - Automatic speech recognition and speech translation
# Model link : https://huggingface.co/openai/whisper-large-v3-turbo
# By default, Transformers uses the sequential algorithm.
# To enable the chunked algorithm, pass the chunk_length_s parameter to the pipeline.
# For large-v3, a chunk length of 30-seconds is optimal. To activate batching over long audio files, pass the argument batch_size


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


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder_attention_mask = torch.ones((1, 1))

    def forward(self, decoder_input_ids, encoder_hidden_states):
        dec_out = self.model.model.decoder(
            decoder_input_ids,
            self.decoder_attention_mask,
            encoder_hidden_states,
        )
        lin_out = self.model.proj_out(dec_out[0])
        return lin_out


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", [ModelVariant.WHISPER_LARGE_V3_TURBO])
def test_whisper_large_v3_speech_translation(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.WHISPER,
        variant=variant,
        task=Task.SPEECH_TRANSLATE,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    input_features = loader.load_inputs()
    framework_model = Wrapper(framework_model)

    # Get decoder inputs
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    encoder_outputs = framework_model.model.model.encoder(input_features)[0].detach()
    encoder_outputs = encoder_outputs.to(torch.float32)
    inputs = [decoder_input_ids, encoder_outputs]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
