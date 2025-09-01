# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Demo - Conditional Generation
# Example of ASR pipeline: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/tests/pipelines/test_pipelines_automatic_speech_recognition.py#L695


import pytest
import torch
from loguru import logger

import forge
from forge.verify.verify import verify
from third_party.tt_forge_models.whisper.pytorch.loader import (
    ModelLoader as WhisperLoader,
)
from third_party.tt_forge_models.whisper.pytorch.loader import (
    ModelVariant as WhisperVariant,
)

variants = [
    pytest.param(
        WhisperVariant.WHISPER_SMALL,
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_linear(variant):

    # Load model and inputs using loader
    loader = WhisperLoader(variant=variant)
    model = loader.load_model()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.proj_out

        def forward(self, ip):

            output = self.model(ip)
            return output

    framework_model = Wrapper(model)

    logger.info("model={}", framework_model)

    inputs = [torch.load("ip.pt")]

    logger.info("inputs={}", inputs)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name="s1")

    # Model Verification
    verify(inputs, framework_model, compiled_model)
