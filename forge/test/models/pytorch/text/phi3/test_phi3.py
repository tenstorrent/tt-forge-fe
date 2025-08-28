# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.phi3.causal_lm.pytorch import (
    ModelLoader as CausalLoader,
)
from third_party.tt_forge_models.phi3.causal_lm.pytorch import (
    ModelVariant as CausalVariant,
)
from third_party.tt_forge_models.phi3.seq_cls.pytorch import ModelLoader as SeqClsLoader
from third_party.tt_forge_models.phi3.seq_cls.pytorch import (
    ModelVariant as SeqClsVariant,
)
from third_party.tt_forge_models.phi3.token_cls.pytorch import (
    ModelLoader as TokenClsLoader,
)
from third_party.tt_forge_models.phi3.token_cls.pytorch import (
    ModelVariant as TokenClsVariant,
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

causal_variants = [
    CausalVariant.MINI_4K,
    CausalVariant.MINI_128K,
]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", causal_variants)
def test_phi3_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3,
        variant=variant.value,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")

    loader = CausalLoader(variant=variant)
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    inputs = loader.load_inputs()

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)
    verify(inputs, framework_model, compiled_model)


token_variants = [
    TokenClsVariant.MINI_4K,
    TokenClsVariant.MINI_128K,
]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", token_variants)
def test_phi3_token_classification(variant):

    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3,
        variant=variant.value,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    loader = TokenClsLoader(variant=variant)
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    inputs = loader.load_inputs()

    compiled_model = forge.compile(framework_model, inputs, module_name)
    verify(inputs, framework_model, compiled_model)


seq_variants = [
    SeqClsVariant.MINI_4K,
    SeqClsVariant.MINI_128K,
]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", seq_variants)
def test_phi3_sequence_classification(variant):

    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3,
        variant=variant.value,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Requires multi-chip support")

    loader = SeqClsLoader(variant=variant)
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    inputs = loader.load_inputs()

    compiled_model = forge.compile(framework_model, inputs, module_name)
    verify(inputs, framework_model, compiled_model)
