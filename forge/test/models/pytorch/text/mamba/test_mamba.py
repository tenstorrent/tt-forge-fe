# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/state-spaces/mamba-2.8b-hf

import pytest
from third_party.tt_forge_models.mamba.pytorch.loader import ModelLoader, ModelVariant

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

variants = [
    pytest.param(
        ModelVariant.MAMBA_790M,
    ),
    pytest.param(
        ModelVariant.MAMBA_2_8B,
        marks=[
            pytest.mark.skip_model_analysis,
        ],
    ),
    pytest.param(
        ModelVariant.MAMBA_1_4B,
        marks=[
            pytest.mark.skip_model_analysis,
        ],
    ),
    pytest.param(
        ModelVariant.MAMBA_370M,
    ),
]


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mamba(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MAMBA,
        variant=variant.value,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )
    if variant in [ModelVariant.MAMBA_2_8B, ModelVariant.MAMBA_1_4B]:
        pytest.xfail(reason="Requires multi-chip support")
    elif variant in [ModelVariant.MAMBA_790M, ModelVariant.MAMBA_370M]:
        pytest.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2586")

    # Load model and inputs using model loader
    model_loader = ModelLoader(variant)
    framework_model = model_loader.load_model()
    inputs_dict = model_loader.load_inputs()

    inputs = [inputs_dict["input_ids"]]
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
        max_new_tokens=100,
        model=compiled_model,
        inputs=padded_inputs,
        seq_len=seq_len,
        tokenizer=model_loader.tokenizer,
    )

    print(generated_text)
