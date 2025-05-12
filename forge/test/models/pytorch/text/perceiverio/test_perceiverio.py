# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/deepmind/language-perceiver

import pytest
from transformers import PerceiverForMaskedLM, PerceiverTokenizer

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize("variant", ["deepmind/language-perceiver"])
def test_perceiverio_masked_lm_pytorch(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="perceiverio",
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

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
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # post processing
    logits = co_out[0]
    masked_tokens_predictions = logits[0, 51:61].argmax(dim=-1)
    print("The predicted token for the [MASK] is: ", tokenizer.decode(masked_tokens_predictions))
