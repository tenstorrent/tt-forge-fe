# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/state-spaces/mamba-2.8b-hf

import pytest
from transformers import AutoTokenizer, MambaForCausalLM

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import generate_no_cache, pad_inputs
from test.utils import download_model

variants = [
    pytest.param(
        "state-spaces/mamba-790m-hf",
        marks=pytest.mark.skip(
            reason="Segmentation fault. Issue Link: https://github.com/tenstorrent/tt-forge-fe/issues/2586"
        ),
    ),
    pytest.param(
        "state-spaces/mamba-2.8b-hf",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 24 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "state-spaces/mamba-1.4b-hf",
        marks=[
            pytest.mark.skip(
                reason="Insufficient host DRAM to run this model (requires a bit more than 24 GB during compile time)"
            ),
            pytest.mark.out_of_memory,
        ],
    ),
    pytest.param(
        "state-spaces/mamba-370m-hf",
        marks=pytest.mark.skip(
            reason="Segmentation fault. Issue Link: https://github.com/tenstorrent/tt-forge-fe/issues/2586"
        ),
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mamba(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MAMBA,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(MambaForCausalLM.from_pretrained, variant, use_cache=False, return_dict=False)
    framework_model.eval()

    # Prepare input
    prompt = "Hey how are you doing?"
    inputs = [tokenizer(prompt, return_tensors="pt")["input_ids"]]
    padded_inputs, seq_len = pad_inputs(*inputs, max_new_tokens=100)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model)

    generated_text = generate_no_cache(
        max_new_tokens=100, model=compiled_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=tokenizer
    )

    print(generated_text)
