# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Demo - Conditional Generation
# Example of ASR pipeline: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/tests/pipelines/test_pipelines_automatic_speech_recognition.py#L695


import pytest
import torch
from loguru import logger

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from third_party.tt_forge_models.whisper.pytorch.loader import (
    ModelLoader as WhisperLoader,
)
from third_party.tt_forge_models.whisper.pytorch.loader import (
    ModelVariant as WhisperVariant,
)

from test.models.models_utils import pad_inputs

variants = [
    pytest.param(
        WhisperVariant.WHISPER_SMALL,
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_whisper(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.WHISPER,
        variant=variant.value,
        task=Task.SPEECH_RECOGNITION,
        source=Source.HUGGINGFACE,
    )
    if variant == WhisperVariant.WHISPER_LARGE:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs using loader
    loader = WhisperLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Get decoder inputs
    decoder_start_token_tensor = torch.tensor(model.generation_config.decoder_start_token_id, dtype=torch.long)
    decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor

    padded_decoder_input_ids, seq_len = pad_inputs(decoder_input_ids, max_new_tokens=100)

    inputs = [inputs, padded_decoder_input_ids]

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features, decoder_input_ids):
            inputs = {"input_features": input_features, "decoder_input_ids": decoder_input_ids}
            output = self.model(**inputs)
            return output

    logger.info("model={}", model)

    framework_model = Wrapper(model)

    # op = framework_model(*inputs)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
