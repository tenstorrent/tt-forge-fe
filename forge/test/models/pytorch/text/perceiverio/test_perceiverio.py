# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/deepmind/language-perceiver

import pytest
from third_party.tt_forge_models.perceiver.pytorch import ModelLoader, ModelVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

variants = [ModelVariant.LANGUAGE_PERCEIVER]


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize("variant", variants)
def test_perceiverio_masked_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PERCEIVERIO,
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    input_dict = loader.load_inputs()

    # mask " missing.". Note that the model performs much better if the masked span starts with a space.
    input_dict.input_ids[0, 52:61] = loader.tokenizer.mask_token_id

    inputs = [input_dict.input_ids, input_dict.attention_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    predicted_text = loader.decode_output(co_out)
    print("The predicted token for the [MASK] is: ", predicted_text)
