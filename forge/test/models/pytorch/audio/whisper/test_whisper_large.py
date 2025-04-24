# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor, WhisperModel

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.utils import download_model

variants = ["openai/whisper-large-v3"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.xfail
def test_whisper_large_v3(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="whisper",
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P2")

    # model loading
    model = download_model(WhisperModel.from_pretrained, variant, return_dict=False)
    feature_extractor = download_model(AutoFeatureExtractor.from_pretrained, variant)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    input_audio = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = input_audio.input_features
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    inputs = [input_features, decoder_input_ids]

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features, decoder_input_ids):
            inputs = {"input_features": input_features, "decoder_input_ids": decoder_input_ids}
            output = self.model(**inputs)
            return output

    framework_model = Wrapper(model)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
