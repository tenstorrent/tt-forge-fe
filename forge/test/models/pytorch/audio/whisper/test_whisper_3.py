# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Large V3 turbo - Automatic speech recognition and speech translation
# Model link : https://huggingface.co/openai/whisper-large-v3-turbo
# By default, Transformers uses the sequential algorithm.
# To enable the chunked algorithm, pass the chunk_length_s parameter to the pipeline.
# For large-v3, a chunk length of 30-seconds is optimal. To activate batching over long audio files, pass the argument batch_size

import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
import forge
from test.models.utils import build_module_name
from forge.verify.verify import verify, VerifyConfig


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
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", ["openai/whisper-large-v3-turbo"])
def test_whisper_large_v3_speech_translation(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="whisper", variant=variant)

    record_forge_property("module_name", module_name)

    processor = WhisperProcessor.from_pretrained(variant)
    framework_model = WhisperForConditionalGeneration.from_pretrained(variant)
    model_config = WhisperConfig.from_pretrained(variant)
    model = Wrapper(framework_model)

    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]
    inputs = processor(sample_audio, return_tensors="pt", sampling_rate=16000)
    input_features = inputs.input_features

    # Get decoder inputs
    decoder_input_ids = torch.tensor([[1, 1]]) * model_config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(torch.int32)
    encoder_outputs = model.model.model.encoder(input_features)[0].detach()
    encoder_outputs = encoder_outputs.to(torch.float32)
    data_input = [decoder_input_ids, encoder_outputs]

    # Compiler test
    compiled_model = forge.compile(model, sample_inputs=data_input, module_name=module_name)

    verify(data_input, model, compiled_model, VerifyConfig(verify_allclose=False))
