# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Demo - Conditional Generation
# Example of ASR pipeline: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/tests/pipelines/test_pipelines_automatic_speech_recognition.py#L695


import pytest
import torch
from third_party.tt_forge_models.whisper.pytorch.loader import (
    ModelLoader as WhisperLoader,
)
from third_party.tt_forge_models.whisper.pytorch.loader import (
    ModelVariant as WhisperVariant,
)

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

variants = [
    pytest.param(
        WhisperVariant.WHISPER_TINY,
        marks=[
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2746"),
        ],
    ),
    pytest.param(
        WhisperVariant.WHISPER_BASE,
        marks=[
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2746"),
        ],
    ),
    pytest.param(
        WhisperVariant.WHISPER_SMALL,
        marks=[
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2746"),
        ],
    ),
    pytest.param(
        WhisperVariant.WHISPER_MEDIUM,
        marks=[
            pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2746"),
        ],
    ),
    pytest.param(
        WhisperVariant.WHISPER_LARGE,
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
        variant=variant.value,
        task=Task.AUDIO_SPEECH_RECOGNITION,
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
            return output.logits

    framework_model = Wrapper(model)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    pcc = 0.99
    if variant in [WhisperVariant.WHISPER_BASE, WhisperVariant.WHISPER_SMALL]:
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
        tokenizer=loader.processor.tokenizer,
    )
    print(generated_text)
