# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify

from test.models.pytorch.text.gemma.utils.model_utils import (
    generate_no_cache,
    pad_inputs,
)
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "google/gemma-1.1-2b-it",
            marks=pytest.mark.push,
        ),
        pytest.param(
            "google/gemma-1.1-7b-it",
            marks=pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 50 GB during compile time)"
            ),
        ),
    ],
)
def test_gemma_pytorch_v1(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="gemma", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority_2")
    forge_property_recorder.record_model_name(module_name)

    # Load model and tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant)
    framework_model = AutoModelForCausalLM.from_pretrained(variant, return_dict=False, use_cache=False)
    framework_model.eval()
    prompt = "What is the tallest mountain?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    max_new_tokens = 200
    padded_inputs, seq_len = pad_inputs(input_ids, max_new_tokens)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Runtime and Post-Processing
    generated_text = generate_no_cache(
        max_new_tokens=max_new_tokens, model=compiled_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=tokenizer
    )
    print(generated_text)
