# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
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
            marks=pytest.mark.xfail(
                reason="[TVM Relay IRModule Generation] aten::bernoulli operator is not implemented"
            ),
        ),
    ],
)
def test_speecht5_tts(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="speecht5_tts",
        variant=variant,
        task=Task.TEXT_TO_SPEECH,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load model and Processer
    processor = download_model(SpeechT5Processor.from_pretrained, variant)
    model = download_model(SpeechT5ForTextToSpeech.from_pretrained, variant, return_dict=False, use_cache=False)
    model.eval()
    framework_model = Wrapper(model)

    # Prepare input
    model_inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")
    decoder_input_values = torch.zeros((1, 1, model.config.num_mel_bins))
    inputs = [model_inputs["input_ids"], model_inputs["attention_mask"], decoder_input_values]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
