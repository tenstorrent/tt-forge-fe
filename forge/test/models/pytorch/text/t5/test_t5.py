# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Task, build_module_name
from test.utils import download_model

variants = [
    pytest.param("t5-small", id="t5-small", marks=[pytest.mark.push]),
    pytest.param(
        "t5-base",
        id="t5-base",
        marks=[pytest.mark.push, pytest.mark.xfail(reason="PCC error due to Reduce Avg data mismatch.")],
    ),
    pytest.param(
        "t5-large", id="t5-large", marks=[pytest.mark.xfail(reason="PCC error due to Reduce Avg data mismatch.")]
    ),
    pytest.param(
        "google/flan-t5-small",
        id="google_flan_t5_small",
        marks=[pytest.mark.push],
    ),
    pytest.param(
        "google/flan-t5-base",
        id="google_flan_t5_base",
    ),
    pytest.param("google/flan-t5-large", id="google_flan_t5_large"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_t5_generation(record_forge_property, variant):
    if variant not in {"t5-small", "google/flan-t5-small", "t5-base", "t5-large"}:
        pytest.skip(f"Skipping {variant} due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="t5", variant=variant, task=Task.TEXT_GENERATION)

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load tokenizer and model from HuggingFace
    # Variants: t5-small, t5-base, t5-large

    config = download_model(T5Config.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = T5Config(**config_dict)
    model = download_model(T5ForConditionalGeneration.from_pretrained, variant, config=config)

    # Wrapper to get around attention mask
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, decoder_input_ids, encoder_outputs):
            return self.model(None, None, decoder_input_ids, None, None, None, None, (encoder_outputs,))

    decoder_input_ids = torch.randint(0, model.config.vocab_size, (1, 1), dtype=torch.int32)
    if "t5-small" in variant:
        encoder_outputs = torch.randn(1, 1, 512)
    elif "t5-base" in variant:
        encoder_outputs = torch.randn(1, 256, 768)
    elif "t5-large" in variant:
        encoder_outputs = torch.randn(1, 256, 1024)

    framework_model = Wrapper(model)

    inputs = [decoder_input_ids, encoder_outputs]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
