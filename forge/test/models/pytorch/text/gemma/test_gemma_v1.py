# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.text.gemma.model_utils.model_utils import (
    generate_no_cache,
    pad_inputs,
)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "google/gemma-1.1-2b-it",
            marks=pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 21 GB during compile time)"
            ),
        ),
        pytest.param(
            "google/gemma-1.1-7b-it",
            marks=pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 31 GB during compile time)"
            ),
        ),
    ],
)
def test_gemma_pytorch_v1(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GEMMA,
        variant=variant,
        task=Task.QA,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
    )

    # Load model and tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant)
    framework_model = AutoModelForCausalLM.from_pretrained(variant, return_dict=False, use_cache=False)
    framework_model.eval()
    prompt = "What is the tallest mountain?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    max_new_tokens = 100
    padded_inputs, seq_len = pad_inputs(input_ids, max_new_tokens)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model)

    # Runtime and Post-Processing
    generated_text = generate_no_cache(
        max_new_tokens=max_new_tokens, model=compiled_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=tokenizer
    )
    print(generated_text)
