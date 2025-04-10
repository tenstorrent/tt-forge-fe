# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Demo - Conditional Generation
# Example of ASR pipeline: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/tests/pipelines/test_pipelines_automatic_speech_recognition.py#L695


import pytest
import torch
from transformers import AutoProcessor, WhisperConfig, WhisperForConditionalGeneration

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.utils import download_model

variants = [
    pytest.param(
        "openai/whisper-tiny",
        marks=[pytest.mark.xfail],
    ),
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_whisper(forge_property_recorder, variant):
    if variant != "openai/whisper-tiny":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="whisper",
        variant=variant,
        task=Task.SPEECH_RECOGNITION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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
    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    input_features = inputs.input_features

    # Get decoder inputs
    decoder_start_token_tensor = torch.tensor(model.generation_config.decoder_start_token_id, dtype=torch.long)
    decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor

    inputs = [input_features, decoder_input_ids]

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
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    current_decoder_input_ids = decoder_input_ids
    all_decoded_ids = decoder_input_ids

    # The iteration count in for _ in range(1) is deliberately limited to 1 to prevent shape mismatches.
    # The model has been compiled specifically for the first decoding step, where decoder_input_ids
    # has a fixed length of (1,1) (the initial token). However, in generative models like Whisper, the length of
    # decoder_input_ids increases with each decoding step as tokens are appended to the sequence.
    # This dynamic increase in shape is incompatible with the static shape expected by the compiled model,
    # leading to a runtime error if subsequent iterations are attempted.

    for _ in range(1):

        # Inference
        outputs = compiled_model(input_features, current_decoder_input_ids)
        logits = outputs[0]

        # Get the next token ID (greedy decoding)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

        # Break if EOS token is generated
        if next_token.item() == model.generation_config.eos_token_id:
            break

        # Append next token to sequence
        all_decoded_ids = torch.cat([all_decoded_ids, next_token], dim=-1)

        # Update decoder inputs for the next iteration
        current_decoder_input_ids = all_decoded_ids

    print("summary : ", processor.decode(all_decoded_ids[0], skip_special_tokens=True))
