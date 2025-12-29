# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.nanogpt.pytorch.loader import ModelLoader, ModelVariant
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


class GPTModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.model.wte(input_ids)
        past_key_values_length = 0
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
        )
        output = self.model(attention_mask=causal_attention_mask, inputs_embeds=inputs_embeds)
        return output


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        ModelVariant.FINANCIAL_SUPPORT_NANOGPT,
    ],
)
def test_nanogpt_text_generation(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.NANOGPT,
        variant=variant.value,
        task=Task.NLP_CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs using model loader
    model_loader = ModelLoader(variant)
    model = model_loader.load_model()
    inputs = model_loader.load_inputs()
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    inputs = [input_ids, attn_mask]

    framework_model = GPTModelWrapper(model)
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
