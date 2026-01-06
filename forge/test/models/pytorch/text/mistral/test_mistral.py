# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from third_party.tt_forge_models.mistral.pytorch import ModelLoader, ModelVariant

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

variants = [ModelVariant.MISTRAL_7B]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mistral(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MISTRAL,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    framework_model.config.use_cache = False
    framework_model.config.sliding_window = None
    framework_model.eval()
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [ModelVariant.MISTRAL_7B_INSTRUCT_V03]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mistral_v0_3(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MISTRAL,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load tokenizer and model
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    framework_model.config.use_cache = False
    framework_model.eval()
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = ["mistralai/Mistral-Nemo-Instruct-2407"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mistral_Nemo(variant):

    # Record Forge Property
    record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MISTRAL,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")


variants = ["mistralai/Mistral-Small-24B-Instruct-2501"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mistral_small_24b(variant):

    # Record Forge Property
    record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MISTRAL,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
        priority=ModelPriority.P1,
        group=ModelGroup.RED,
    )

    # Force the test to fail explicitly
    pytest.xfail(reason="Requires multi-chip support")


variants = ["mistralai/Mistral-Large-Instruct-2411"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mistral_large(variant):

    # Record Forge Property
    record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MISTRAL,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
        priority=ModelPriority.P1,
        group=ModelGroup.RED,
    )

    # Force the test to fail explicitly
    pytest.xfail(reason="Requires multi-chip support")
