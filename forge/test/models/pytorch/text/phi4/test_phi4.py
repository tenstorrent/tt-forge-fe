# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.phi4.causal_lm.pytorch.loader import (
    ModelLoader as CausalLMLoader,
)
from third_party.tt_forge_models.phi4.seq_cls.pytorch.loader import (
    ModelLoader as SequenceClassificationLoader,
)
from third_party.tt_forge_models.phi4.token_cls.pytorch.loader import (
    ModelLoader as TokenClassificationLoader,
)

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

variants = ["microsoft/phi-4"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi_4_causal_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI4,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load model using the new loader
    loader = CausalLMLoader()
    model = loader.load_model()

    # Wrap the model as expected by the test
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # Generate inputs using the loader
    sample_inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs, module_name)

    # Model Verification
    verify(sample_inputs, framework_model, compiled_model)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail
def test_phi_4_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI4,
        variant=variant,
        task=Task.NLP_TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Test is killed at consteval compilation stage")

    # Load model using the new loader
    loader = TokenClassificationLoader()
    model = loader.load_model()

    # Wrap the model as expected by the test
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # Generate inputs using the loader
    inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_phi_4_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI4,
        variant=variant,
        task=Task.NLP_SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Test is killed at consteval compilation stage")

    # Load model using the new loader
    loader = SequenceClassificationLoader()
    model = loader.load_model()

    # Wrap the model as expected by the test
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # Generate inputs using the loader
    inputs = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
