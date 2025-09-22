# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from third_party.tt_forge_models.gemma.pytorch import ModelLoader as CausalLMLoader
from third_party.tt_forge_models.gemma.pytorch import ModelVariant as CausalLMVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import TextModelWrapper

variants = [
    CausalLMVariant.GEMMA_2B,
]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=[v.value for v in variants])
def test_gemma_2b(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GEMMA,
        variant=variant.value,
        source=Source.HUGGINGFACE,
        task=Task.TEXT_GENERATION,
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Use loader for model and inputs
    loader = CausalLMLoader(variant)
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(CausalLMVariant.GEMMA_2_2B_IT, marks=pytest.mark.xfail),
        pytest.param(CausalLMVariant.GEMMA_2_9B_IT, marks=[pytest.mark.xfail, pytest.mark.out_of_memory]),
    ],
)
def test_gemma_pytorch_v2(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GEMMA,
        variant=variant.value,
        task=Task.QA,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )
    if variant == CausalLMVariant.GEMMA_2_9B_IT:
        pytest.xfail(reason="Requires multi-chip support")
    elif variant == CausalLMVariant.GEMMA_2_2B_IT:
        pytest.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2844")

    # Load model and input via loader
    loader = CausalLMLoader(variant)
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()
    input_dict = loader.load_inputs(max_new_tokens=200, prompt="What is the tallest mountain?")
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    CausalLMVariant.GEMMA_2_27B_IT,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_gemma_pytorch_27b(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.FLUX,
        variant=variant.value,
        task=Task.QA,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")
