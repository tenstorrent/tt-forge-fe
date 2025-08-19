# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from third_party.tt_forge_models.gpt2.pytorch import ModelLoader, ModelVariant

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

VARIANT_TO_TASK = {
    ModelVariant.GPT2_BASE: Task.TEXT_GENERATION,
    ModelVariant.GPT2_SEQUENCE_CLASSIFICATION: Task.SEQUENCE_CLASSIFICATION,
}

VARIANTS = [
    ModelVariant.GPT2_BASE,
    ModelVariant.GPT2_SEQUENCE_CLASSIFICATION,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", VARIANTS)
def test_gpt2_variants(variant):
    # Determine task
    task = VARIANT_TO_TASK[variant]

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GPT,
        variant=variant,
        task=task,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and inputs via ModelLoader
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs_dict = loader.load_inputs()
    inputs = [inputs_dict["input_ids"]]

    # Compile with Forge
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)

    # Run verification
    _, co_out = verify(inputs, model, compiled_model)

    # For classification variant, decode and print sentiment
    if variant == ModelVariant.GPT2_SEQUENCE_CLASSIFICATION:
        predicted_value = loader.decode_output(co_out)
        print(f"Predicted Sentiment: {predicted_value}")
