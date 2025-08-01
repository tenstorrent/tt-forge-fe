# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# CodeGen Demo - CasualLM

import pytest
from third_party.tt_forge_models.codegen.pytorch.loader import ModelLoader, ModelVariant

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

variants = [
    ModelVariant.CODEGEN_350M_MONO,
    ModelVariant.CODEGEN_350M_MULTI,
    ModelVariant.CODEGEN_350M_NL,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_codegen(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.CODEGEN,
        variant=variant.value,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs using model loader
    model_loader = ModelLoader(variant)
    model = model_loader.load_model()
    framework_model = TextModelWrapper(model=model, text_embedding=model.transformer.wte)
    framework_model.eval()
    inputs_dict = model_loader.load_inputs()
    input_ids = inputs_dict["input_ids"]
    attn_mask = inputs_dict["attention_mask"]
    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
