# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Demo - Conditional Generation
# Example of ASR pipeline: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/tests/pipelines/test_pipelines_automatic_speech_recognition.py#L695


import pytest
import torch
from transformers import AutoProcessor, WhisperConfig, WhisperForConditionalGeneration

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import AutomaticValueChecker, VerifyConfig
from forge.verify.verify import verify

from test.models.models_utils import (
    generate_no_cache_for_encoder_decoder_model,
    pad_inputs,
)
from test.utils import download_model

variants = [
    pytest.param(
        "openai/whisper-tiny",
    ),
    pytest.param(
        "openai/whisper-base",
    ),
    pytest.param(
        "openai/whisper-small",
        marks=pytest.mark.xfail(
            reason="Data mismatch. PCC = 0.9498597935454118, but required = 0.99. Issue link: https://github.com/tenstorrent/tt-forge-fe/issues/2587"
        ),
    ),
    pytest.param(
        "openai/whisper-medium",
    ),
    pytest.param(
        "openai/whisper-large",
        marks=[
            pytest.mark.out_of_memory,
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_whisper(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.WHISPER,
        variant=variant,
        task=Task.AUDIO_ASR,
        source=Source.HUGGINGFACE,
    )
    if variant == "openai/whisper-large":
        pytest.xfail(reason="Requires multi-chip support")

    # Load model (with tokenizer and feature extractor)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    model_config = WhisperConfig.from_pretrained(variant)
    model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        config=model_config,
    )
    model.config.use_cache = False

    # Load and preprocess sample audio
    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt", weights_only=False)
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    input_features = inputs.input_features

    # Get decoder inputs
    decoder_start_token_tensor = torch.tensor(model.generation_config.decoder_start_token_id, dtype=torch.long)
    decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
    padded_decoder_input_ids, seq_len = pad_inputs(
        decoder_input_ids, max_new_tokens=100
    )  # Whisper only supports up to 448 decoder positions.

    inputs = [input_features, padded_decoder_input_ids]

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features, decoder_input_ids):
            inputs = {"input_features": input_features, "decoder_input_ids": decoder_input_ids}
            output = self.model(**inputs)
            return output.logits

    framework_model = Wrapper(model)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    pcc = 0.99
    if variant == "openai/whisper-base":
        pcc = 0.95

    # Model Verification
    verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    generated_text = generate_no_cache_for_encoder_decoder_model(
        max_new_tokens=100,
        model=compiled_model,
        input_ids=inputs[0],
        decoder_input_ids=padded_decoder_input_ids,
        seq_len=seq_len,
        tokenizer=processor.tokenizer,
    )
    print(generated_text)
