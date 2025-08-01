# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.gpt_neo.causal_lm.pytorch import (
    ModelLoader as CausalLMLoader,
)
from third_party.tt_forge_models.gpt_neo.causal_lm.pytorch.loader import (
    ModelVariant as CausalLMVariant,
)
from third_party.tt_forge_models.gpt_neo.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
)
from third_party.tt_forge_models.gpt_neo.sequence_classification.pytorch.loader import (
    ModelVariant as SequenceClassificationVariant,
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

GPTNEO_VARIANTS = [
    CausalLMVariant.GPT_NEO_125M,
    CausalLMVariant.GPT_NEO_1_3B,
    pytest.param(
        CausalLMVariant.GPT_NEO_2_7B,
        marks=[
            pytest.mark.out_of_memory,
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", GPTNEO_VARIANTS)
def test_gptneo_causal_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GPTNEO,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )
    if variant == CausalLMVariant.GPT_NEO_2_7B:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and input
    loader = CausalLMLoader(variant)
    model = loader.load_model()
    input_dict = loader.load_inputs()

    # prepare input and model
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]
    framework_model = TextModelWrapper(model=model, text_embedding=model.transformer.wte)
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


GPTNEO_SEQ_CLS_VARIANTS = [
    SequenceClassificationVariant.GPT_NEO_125M,
    SequenceClassificationVariant.GPT_NEO_1_3B,
    pytest.param(
        SequenceClassificationVariant.GPT_NEO_2_7B,
        marks=[
            pytest.mark.out_of_memory,
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", GPTNEO_SEQ_CLS_VARIANTS)
def test_gptneo_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GPTNEO,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    if variant == SequenceClassificationVariant.GPT_NEO_2_7B:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and input
    loader = SequenceClassificationLoader(variant)
    model = loader.load_model()
    input_dict = loader.load_inputs()

    # prepare input and model
    inputs = [input_dict["input_ids"]]
    framework_model = TextModelWrapper(model=model)
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)
    # post processing
    print(f"predicted category: {loader.decode_output(co_out)}")
