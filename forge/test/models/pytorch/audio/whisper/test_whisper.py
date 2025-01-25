# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Demo - Conditional Generation
# Example of ASR pipeline: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/tests/pipelines/test_pipelines_automatic_speech_recognition.py#L695


import pytest
import torch
from transformers import AutoProcessor, WhisperConfig, WhisperForConditionalGeneration

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, build_module_name
from test.utils import download_model

variants = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
]


def generate_model_whisper_congen_hf_pytorch(variant):
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

    # Load model (with tokenizer and feature extractor)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    model_config = WhisperConfig.from_pretrained(variant)

    framework_model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        config=model_config,
    )
    framework_model = Wrapper(framework_model)

    # Load and preprocess sample audio
    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    input_features = inputs.input_features

    # Get decoder inputs
    decoder_input_ids = torch.tensor([[1, 1]]) * model_config.decoder_start_token_id
    decoder_input_ids = decoder_input_ids.to(torch.int32)
    encoder_outputs = framework_model.model.model.encoder(input_features)[0].detach()
    encoder_outputs = encoder_outputs.to(torch.float32)

    return framework_model, [decoder_input_ids, encoder_outputs]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_whisper(record_forge_property, variant):
    if variant != "openai/whisper-tiny":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="whisper", variant=variant)

    # Record Forge Property
    record_forge_property("model_name", module_name)

    framework_model, inputs = generate_model_whisper_congen_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
