# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Llama3 Demo - CasualLM

import pytest
from third_party.tt_forge_models.llama.causal_lm.pytorch import (
    ModelLoader as CausalLMLoader,
)
from third_party.tt_forge_models.llama.causal_lm.pytorch import (
    ModelVariant as CausalLMVariant,
)
from third_party.tt_forge_models.llama.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
)
from third_party.tt_forge_models.llama.sequence_classification.pytorch import (
    ModelVariant as SequenceClassificationVariant,
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

llama_loader_variants = [
    pytest.param(
        CausalLMVariant.LLAMA_3_2_1B,
        marks=pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2833"),
    ),
    pytest.param(
        CausalLMVariant.LLAMA_3_2_1B_INSTRUCT,
        marks=pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2833"),
    ),
    pytest.param(
        CausalLMVariant.LLAMA_3_8B,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        CausalLMVariant.LLAMA_3_8B_INSTRUCT,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        CausalLMVariant.LLAMA_3_1_8B,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        CausalLMVariant.LLAMA_3_1_8B_INSTRUCT,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        CausalLMVariant.LLAMA_3_2_3B,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        CausalLMVariant.LLAMA_3_2_3B_INSTRUCT,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        CausalLMVariant.HUGGYLLAMA_7B,
        marks=[pytest.mark.out_of_memory],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", llama_loader_variants)
def test_llama3_causal_lm_pytorch(variant):
    if variant in [
        CausalLMVariant.LLAMA_3_1_8B_INSTRUCT,
        CausalLMVariant.LLAMA_3_2_1B_INSTRUCT,
        CausalLMVariant.LLAMA_3_2_3B_INSTRUCT,
        CausalLMVariant.LLAMA_3_1_70B,
        CausalLMVariant.LLAMA_3_1_70B_INSTRUCT,
        CausalLMVariant.LLAMA_3_3_70B_INSTRUCT,
        CausalLMVariant.LLAMA_3_8B_INSTRUCT,
    ]:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.LLAMA3,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=group,
        priority=priority,
    )

    if variant not in [
        CausalLMVariant.LLAMA_3_2_1B,
        CausalLMVariant.LLAMA_3_2_1B_INSTRUCT,
    ]:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and input
    loader = CausalLMLoader(variant)
    model = loader.load_model()
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()
    input_dict, seq_len = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)


llama_seq_cls_variants = [
    pytest.param(SequenceClassificationVariant.LLAMA_3_2_1B),
    pytest.param(SequenceClassificationVariant.LLAMA_3_2_1B_INSTRUCT),
    pytest.param(
        SequenceClassificationVariant.LLAMA_3_8B,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        SequenceClassificationVariant.LLAMA_3_8B_INSTRUCT,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        SequenceClassificationVariant.LLAMA_3_1_8B,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        SequenceClassificationVariant.LLAMA_3_1_8B_INSTRUCT,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        SequenceClassificationVariant.LLAMA_3_2_3B,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        SequenceClassificationVariant.LLAMA_3_2_3B_INSTRUCT,
        marks=[pytest.mark.out_of_memory],
    ),
    pytest.param(
        SequenceClassificationVariant.HUGGYLLAMA_7B,
        marks=[pytest.mark.out_of_memory],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", llama_seq_cls_variants)
def test_llama3_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.LLAMA3,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    if variant not in [
        SequenceClassificationVariant.LLAMA_3_2_1B,
        SequenceClassificationVariant.LLAMA_3_2_1B_INSTRUCT,
    ]:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and input
    loader = SequenceClassificationLoader(variant)
    framework_model = loader.load_model()
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"]]
    framework_model = TextModelWrapper(model=framework_model)
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    print(f"Prediction: {loader.decode_output(co_out)}")
