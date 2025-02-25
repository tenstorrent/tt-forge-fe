# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/deepmind/language-perceiver

import pytest
from transformers import PerceiverForMaskedLM, PerceiverTokenizer

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize("variant", ["deepmind/language-perceiver"])
def test_perceiverio_masked_lm_pytorch(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="perceiverio",
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    # Load model and tokenizer
    tokenizer = download_model(PerceiverTokenizer.from_pretrained, variant)
    framework_model = download_model(PerceiverForMaskedLM.from_pretrained, variant, return_dict=False)
    framework_model.eval()

    # Prepare input
    text = "This is an incomplete sentence where some words are missing."
    encoding = tokenizer(text, padding="max_length", return_tensors="pt")
    # mask " missing.". Note that the model performs much better if the masked span starts with a space.
    encoding.input_ids[0, 52:61] = tokenizer.mask_token_id

    inputs = [encoding.input_ids, encoding.attention_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    # Inference
    output = compiled_model(*inputs)

    # post processing
    logits = output[0]
    masked_tokens_predictions = logits[0, 51:61].argmax(dim=-1)
    print("The predicted token for the [MASK] is: ", tokenizer.decode(masked_tokens_predictions))
