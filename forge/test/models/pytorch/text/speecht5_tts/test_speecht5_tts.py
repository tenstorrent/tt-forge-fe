# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import types

import pytest
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

import forge
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.models_utils import embed_pos
from test.utils import download_model


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_values):
        return self.model(input_ids, attention_mask, decoder_input_values)[0]  # Return only the spectrogram tensor


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/speecht5_tts",
            marks=[pytest.mark.xfail],
        ),
    ],
)
def test_speecht5_tts(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="speecht5_tts",
        variant=variant,
        task=Task.TEXT_TO_SPEECH,
        source=Source.HUGGINGFACE,
    )

    # Load model and Processer
    processor = download_model(SpeechT5Processor.from_pretrained, variant)
    model = download_model(SpeechT5ForTextToSpeech.from_pretrained, variant, return_dict=False, use_cache=False)
    model.speecht5.encoder.wrapped_encoder.embed_positions.forward = types.MethodType(
        embed_pos, model.speecht5.encoder.wrapped_encoder.embed_positions
    )
    model.eval()
    framework_model = Wrapper(model)

    # Prepare input
    model_inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")
    decoder_input_values = torch.zeros((1, 1, model.config.num_mel_bins))
    inputs = [model_inputs["input_ids"], model_inputs["attention_mask"], decoder_input_values]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
