# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Large V3 turbo - Automatic speech recognition and speech translation
# Model link : https://huggingface.co/openai/whisper-large-v3-turbo
# By default, Transformers uses the sequential algorithm.
# To enable the chunked algorithm, pass the chunk_length_s parameter to the pipeline.
# For large-v3, a chunk length of 30-seconds is optimal. To activate batching over long audio files, pass the argument batch_size

import pytest
import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name


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
@pytest.mark.parametrize("variant", ["openai/whisper-large-v3-turbo"])
def test_whisper_large_v3_speech_translation(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="whisper",
        variant=variant,
        task=Task.SPEECH_TRANSLATE,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    processor = WhisperProcessor.from_pretrained(variant)
    framework_model = WhisperForConditionalGeneration.from_pretrained(variant)
    model_config = WhisperConfig.from_pretrained(variant)
    framework_model = Wrapper(framework_model)

    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]
    processed_inputs = processor(sample_audio, return_tensors="pt", sampling_rate=16000)
    input_features = processed_inputs.input_features

    # Get decoder inputs
    decoder_input_ids = torch.tensor([[1, 1]]) * model_config.decoder_start_token_id
    encoder_outputs = framework_model.model.model.encoder(input_features)[0].detach()
    encoder_outputs = encoder_outputs.to(torch.float32)
    inputs = [decoder_input_ids, encoder_outputs]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
