# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import (
    generate_no_cache_for_encoder_decoder_model,
    pad_inputs,
)
from test.utils import download_model

variants = [
    pytest.param(
        "t5-small",
        id="t5-small",
    ),
    pytest.param(
        "t5-base",
        id="t5-base",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "t5-large",
        id="t5-large",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "google/flan-t5-small",
        id="google_flan_t5_small",
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        "google/flan-t5-base",
        id="google_flan_t5_base",
    ),
    pytest.param("google/flan-t5-large", id="google_flan_t5_large"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_t5_generation(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.T5,
        variant=variant,
        task=Task.NLP_TEXT_GEN,
        source=Source.HUGGINGFACE,
    )
    if variant not in ["t5-small", "google/flan-t5-small", "t5-base", "t5-large"]:
        pytest.xfail(reason="Requires multi-chip support")

    # Load tokenizer and model from HuggingFace
    # Variants: t5-small, t5-base, t5-large

    config = download_model(T5Config.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = T5Config(**config_dict)
    model = download_model(T5ForConditionalGeneration.from_pretrained, variant, config=config)
    tokenizer = AutoTokenizer.from_pretrained(variant)

    inputs = tokenizer(
        "summarize: Researchers have extensively studied the benefits of having pets, "
        "particularly dogs, on human health and well-being. Findings suggest that pet ownership "
        "can lead to improved mental health, reduced stress levels, and even physical health benefits "
        "such as lower blood pressure and increased physical activity levels due to regular walks.",
        return_tensors="pt",
    )

    input_ids = inputs.input_ids
    decoder_start_token_tensor = torch.tensor(model.generation_config.decoder_start_token_id, dtype=torch.long)
    decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
    padded_decoder_input_ids, seq_len = pad_inputs(decoder_input_ids)
    inputs = [input_ids, padded_decoder_input_ids]

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, decoder_input_ids):
            inputs = {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}
            output = self.model(**inputs)
            return output

    framework_model = Wrapper(model)

    # Forge compile
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    generated_text = generate_no_cache_for_encoder_decoder_model(
        max_new_tokens=512,
        model=compiled_model,
        input_ids=inputs[0],
        decoder_input_ids=padded_decoder_input_ids,
        seq_len=seq_len,
        tokenizer=tokenizer,
    )
    print(generated_text)
