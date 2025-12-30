# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from third_party.tt_forge_models.qwen_1_5.causal_lm.pytorch import (
    ModelLoader as CausalLMLoader,
)
from third_party.tt_forge_models.qwen_1_5.causal_lm.pytorch import (
    ModelVariant as CausalLMVariant,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import TextModelWrapper


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            CausalLMVariant.QWEN_1_5_0_5B,
            marks=[pytest.mark.xfail],
        ),
    ],
)
def test_qwen1_5_causal_lm(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWEN15,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs
    loader = CausalLMLoader(variant=variant)
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
@pytest.mark.xfail
@pytest.mark.parametrize("variant", [CausalLMVariant.QWEN_1_5_0_5B_CHAT])
def test_qwen1_5_chat(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWEN15,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs
    loader = CausalLMLoader(variant=variant)
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()

    # Load inputs with chat template (auto-detected from variant name)
    input_dict = loader.load_inputs(batch_size=1)
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
