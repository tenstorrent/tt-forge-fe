# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

variants = [
    pytest.param(
        "t5-small",
        id="t5-small",
    ),
    pytest.param(
        "t5-base",
        id="t5-base",
        marks=[pytest.mark.xfail(reason="Data mismatch -> AutomaticValueChecker (compare_with_golden)")],
    ),
    pytest.param(
        "t5-large",
        id="t5-large",
        marks=[pytest.mark.xfail(reason="Data mismatch -> AutomaticValueChecker (compare_with_golden)")],
    ),
    pytest.param(
        "google/flan-t5-small",
        id="google_flan_t5_small",
        marks=[pytest.mark.xfail(reason="Data mismatch -> AutomaticValueChecker (compare_with_golden)")],
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
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="t5", variant=variant, task=Task.TEXT_GENERATION, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

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
    inputs = [input_ids, decoder_input_ids]

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

    current_decoder_input_ids = decoder_input_ids
    all_decoded_ids = decoder_input_ids

    # The iteration count in for _ in range(1) is deliberately limited to 1 to prevent shape mismatches.
    # The model has been compiled specifically for the first decoding step, where decoder_input_ids
    # has a fixed length of (1,1) (the initial token). However, in generative models like T5, the length of
    # decoder_input_ids increases with each decoding step as tokens are appended to the sequence.
    # This dynamic increase in shape is incompatible with the static shape expected by the compiled model,
    # leading to a runtime error if subsequent iterations are attempted.

    for _ in range(1):

        # Inference
        outputs = compiled_model(input_ids, current_decoder_input_ids)
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

    print("summary : ", tokenizer.decode(all_decoded_ids[0], skip_special_tokens=True))
