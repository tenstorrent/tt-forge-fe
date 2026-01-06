# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from third_party.tt_forge_models.stereo.pytorch import ModelLoader, ModelVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

variants = [
    pytest.param(ModelVariant.SMALL),
    pytest.param(ModelVariant.MEDIUM),
    pytest.param(ModelVariant.LARGE, marks=[pytest.mark.out_of_memory]),
]


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids}
        output = self.model(**inputs)
        return output.logits


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_stereo(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.STEREO,
        variant=variant,
        task=Task.MM_MUSIC_GENERATION,
        source=Source.HUGGINGFACE,
    )
    if variant == ModelVariant.LARGE:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    framework_model = Wrapper(model)
    inputs_dict = loader.load_inputs()
    inputs = [inputs_dict["input_ids"], inputs_dict["attention_mask"], inputs_dict["decoder_input_ids"]]

    # Issue: https://github.com/tenstorrent/tt-forge-fe/issues/615
    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
