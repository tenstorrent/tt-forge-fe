# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask, None, False, False, True)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "microsoft/speecht5_tts",
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

    # Load model and Processer
    processor = download_model(SpeechT5Processor.from_pretrained, variant)
    model = download_model(SpeechT5ForTextToSpeech.from_pretrained, variant, return_dict=False, use_cache=False)
    model.eval()
    framework_model = Wrapper(model)
    framework_model = framework_model.model.speecht5.encoder

    # Prepare input
    input_values = torch.randint(0, 50, (1, 24))
    attention_mask = torch.randint(0, 50, (1, 24))
    inputs = [input_values, attention_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
