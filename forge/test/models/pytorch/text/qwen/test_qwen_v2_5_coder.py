# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.qwen_2_5_coder.pytorch import ModelLoader, ModelVariant

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

# Variants for testing
variants = [
    pytest.param(
        ModelVariant.QWEN_2_5_CODER_0_5B,
        marks=[pytest.mark.xfail],
    ),
    pytest.param(
        ModelVariant.QWEN_2_5_CODER_1_5B,
        marks=[pytest.mark.xfail],
    ),
    # pytest.param(
    #     ModelVariant.QWEN_2_5_CODER_1_5B_INSTRUCT,
    #     marks=[
    #         pytest.mark.out_of_memory,
    #     ],
    # ),
    # pytest.param(
    #     ModelVariant.QWEN_2_5_CODER_3B,
    #     marks=[
    #         pytest.mark.out_of_memory,
    #     ],
    # ),
    # pytest.param(
    #     ModelVariant.QWEN_2_5_CODER_3B_INSTRUCT,
    #     marks=[
    #         pytest.mark.out_of_memory,
    #     ],
    # ),
    # pytest.param(
    #     ModelVariant.QWEN_2_5_CODER_7B,
    #     marks=[
    #         pytest.mark.out_of_memory,
    #     ],
    # ),
    # pytest.param(
    #     ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT,
    #     marks=[
    #         pytest.mark.out_of_memory,
    #     ],
    # ),
    # pytest.param(
    #     ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT,
    #     marks=[pytest.mark.out_of_memory],
    # ),
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.xfail
def test_qwen_coder_clm_pytorch(variant):

    if variant == ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWENCODER,
        variant=variant,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=group,
        priority=priority,
    )

    if variant in [
        ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT,
        ModelVariant.QWEN_2_5_CODER_1_5B_INSTRUCT,
        ModelVariant.QWEN_2_5_CODER_3B,
        ModelVariant.QWEN_2_5_CODER_3B_INSTRUCT,
        ModelVariant.QWEN_2_5_CODER_7B,
        ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT,
    ]:
        pytest.xfail(reason="Requires multi-chip support")

    # Load Model and inputs using loader
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    model.config.use_cache = False
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
