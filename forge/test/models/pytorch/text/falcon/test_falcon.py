# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant
from transformers import AutoTokenizer, FalconForCausalLM

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

from test.models.models_utils import generate_no_cache, pad_inputs


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["tiiuae/falcon-7b-instruct"])
def test_falcon(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.FALCON,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    tokenizer = AutoTokenizer.from_pretrained(variant)
    model = FalconForCausalLM.from_pretrained(variant)
    model.config.use_cache = False
    model.config.return_dict = False

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids, None, attention_mask)

    framework_model = Wrapper(model)
    input_tokens = tokenizer("Hello, my dog is cute", return_tensors="pt")

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    pytest.param(ModelVariant.FALCON_1B),
    pytest.param(
        ModelVariant.FALCON_3B,
        marks=[pytest.mark.xfail, pytest.mark.out_of_memory],
    ),
    pytest.param(
        ModelVariant.FALCON_7B,
        marks=[pytest.mark.xfail, pytest.mark.out_of_memory],
    ),
    pytest.param(
        ModelVariant.FALCON_10B,
        marks=[pytest.mark.xfail, pytest.mark.out_of_memory],
    ),
    pytest.param(
        ModelVariant.FALCON_MAMBA_7B,
        marks=[
            pytest.mark.out_of_memory,
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_falcon_3(variant):
    # Record Forge Property
    if variant in [
        ModelVariant.FALCON_1B,
        ModelVariant.FALCON_3B,
        ModelVariant.FALCON_7B,
        ModelVariant.FALCON_10B,
    ]:
        group = ModelGroup.RED
    else:
        group = ModelGroup.GENERALITY

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.FALCON3,
        variant=variant.value,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=group,
    )

    if variant != ModelVariant.FALCON_1B:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs using model loader
    model_loader = ModelLoader(variant)
    framework_model = model_loader.load_model()
    framework_model.eval()

    # Load inputs and pad them
    inputs = model_loader.load_inputs()
    padded_inputs, seq_len = pad_inputs(inputs)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model)

    # post processing
    generated_text = generate_no_cache(
        max_new_tokens=50, model=compiled_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=model_loader.tokenizer
    )

    print("generated_text : ", generated_text)
