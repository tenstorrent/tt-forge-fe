# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.phi1.causal_lm.pytorch import (
    ModelLoader as CausalLMLoader,
)
from third_party.tt_forge_models.phi1.causal_lm.pytorch.loader import (
    ModelVariant as CausalLMVariant,
)
from third_party.tt_forge_models.phi1.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
)
from third_party.tt_forge_models.phi1.sequence_classification.pytorch.loader import (
    ModelVariant as SequenceClassificationVariant,
)
from third_party.tt_forge_models.phi1.token_classification.pytorch import (
    ModelLoader as TokenClassificationLoader,
)
from third_party.tt_forge_models.phi1.token_classification.pytorch.loader import (
    ModelVariant as TokenClassificationVariant,
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

PHI_VARIANTS = [
    CausalLMVariant.PHI1,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", PHI_VARIANTS)
def test_phi1_causal_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI1,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and input
    loader = CausalLMLoader(variant)
    model = loader.load_model()
    input_dict = loader.load_inputs()

    # prepare input and model
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]
    model.config.use_cache = False
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


PHI_VARIANTS = [
    TokenClassificationVariant.PHI1,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", PHI_VARIANTS)
def test_phi1_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI1,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and input
    loader = TokenClassificationLoader(variant)
    model = loader.load_model()
    input_dict = loader.load_inputs()

    # prepare input and model
    inputs = [input_dict["input_ids"]]
    model.config.use_cache = False
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    print(f"Answer: {loader.decode_output(co_out)}")


PHI_VARIANTS = [
    SequenceClassificationVariant.PHI1,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", PHI_VARIANTS)
def test_phi1_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI1,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and input
    loader = SequenceClassificationLoader(variant)
    model = loader.load_model()
    input_dict = loader.load_inputs()

    # prepare input and model
    inputs = [input_dict["input_ids"]]
    model.config.use_cache = False
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    print(f"Predicted Sentiment: {loader.decode_output(co_out)}")
