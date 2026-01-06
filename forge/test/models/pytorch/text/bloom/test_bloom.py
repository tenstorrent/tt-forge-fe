# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.bloom.pytorch import ModelLoader
from transformers import BloomModel

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import (
    _prepare_4d_causal_attention_mask_with_cache_position,
)

BloomModel._prepare_4d_causal_attention_mask_with_cache_position = _prepare_4d_causal_attention_mask_with_cache_position


# Wrapper to get around past key values
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, None, attention_mask)
        return output


@pytest.mark.nightly
@pytest.mark.xfail
def test_bloom():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BLOOM,
        variant="default",
        source=Source.HUGGINGFACE,
        task=Task.NLP_CAUSAL_LM,
    )

    # Load model and input
    loader = ModelLoader()
    model = loader.load_model()
    model.eval()
    model.config.use_cache = False
    model.config.return_dict = False
    framework_model = Wrapper(model)
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
