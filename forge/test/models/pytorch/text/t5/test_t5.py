# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.t5.pytorch import ModelLoader, ModelVariant

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
    generate_no_cache_for_encoder_decoder_model,
    pad_inputs,
)


class T5Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids):
        inputs = {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}
        output = self.model(**inputs)
        return output


variants = [
    pytest.param(ModelVariant.SMALL),
    pytest.param(ModelVariant.BASE),
    pytest.param(ModelVariant.LARGE, marks=[pytest.mark.xfail]),
    pytest.param(ModelVariant.FLAN_T5_SMALL),
    pytest.param(ModelVariant.FLAN_T5_BASE),
    pytest.param(ModelVariant.FLAN_T5_LARGE, marks=[pytest.mark.xfail]),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_t5_generation(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.T5,
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )

    if variant in [ModelVariant.LARGE, ModelVariant.FLAN_T5_LARGE]:
        pytest.xfail(reason="Fatal Python error")

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    input_dict = loader.load_inputs()

    # Extract the inputs and prepare them for the model
    input_ids = input_dict["input_ids"]
    decoder_input_ids = input_dict["decoder_input_ids"]
    padded_decoder_input_ids, seq_len = pad_inputs(decoder_input_ids)
    inputs = [input_ids, padded_decoder_input_ids]

    # Create wrapper for the model
    framework_model = T5Wrapper(model)

    # Forge compile
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    # Generate text using the compiled model
    generated_text = generate_no_cache_for_encoder_decoder_model(
        max_new_tokens=512,
        model=compiled_model,
        input_ids=inputs[0],
        decoder_input_ids=padded_decoder_input_ids,
        seq_len=seq_len,
        tokenizer=loader.tokenizer,
    )
    print(generated_text)
