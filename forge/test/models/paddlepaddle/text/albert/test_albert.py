# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from forge.forge_property_utils import Framework, Source, Task

from paddlenlp.transformers import AlbertForMaskedLM, AlbertTokenizer

variants = ["albert-chinese-tiny"]
inputs = [["一，[MASK]，三，四"]]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.parametrize("input", inputs)
def test_albert_maskedlm(forge_property_recorder, variant, input):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="albert",
        variant=variant[7:],
        task=Task.MASKED_LM,
        source=Source.PADDLENLP,
    )
    forge_property_recorder.record_group("generality")

    # Load Model and Tokenizer
    model = AlbertForMaskedLM.from_pretrained(variant)
    tokenizer = AlbertTokenizer.from_pretrained(variant)

    # Load sample
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)
    inputs = [paddle.to_tensor(value) for value in encoded_input.values()]

    # Wrap Model to fix input signature
    class AlbertWrapper(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, token_type_ids, position_ids, attention_mask):
            return self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

    model = AlbertWrapper(model)

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        framework_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
