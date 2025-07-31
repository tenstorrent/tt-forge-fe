# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

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
from third_party.tt_forge_models.phi3.pytorch.loader import ModelLoader

from test.models.models_utils import TextModelWrapper

variants = ["microsoft/phi-3-mini-4k-instruct", "microsoft/phi-3-mini-128k-instruct"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs using ModelLoader
    loader = ModelLoader(variant=variant, task="causal_lm")
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # Load inputs (returns [input_ids, attention_mask])
    inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_token_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs using ModelLoader
    loader = ModelLoader(variant=variant, task="token_classification")
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # Load inputs (returns [input_ids])
    inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_sequence_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs using ModelLoader
    loader = ModelLoader(variant=variant, task="sequence_classification")
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # Load inputs (returns [input_ids])
    inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
