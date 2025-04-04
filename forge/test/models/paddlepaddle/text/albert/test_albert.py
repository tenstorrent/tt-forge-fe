# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from test.models.utils import Framework, Source, Task, build_module_name

from paddlenlp.transformers import AlbertForMaskedLM, AlbertTokenizer

variants = ["albert-chinese-tiny"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.parametrize("input", [["你好，我的狗很可爱"]])
def test_albert_maskedlm(forge_property_recorder, variant, input):
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="albert",
        variant=variant[7:],
        task=Task.MASKED_LM,
        source=Source.PADDLENLP,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load Model and Tokenizer
    model = AlbertForMaskedLM.from_pretrained(variant)
    tokenizer = AlbertTokenizer.from_pretrained(variant)

    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)

    inputs = [paddle.to_tensor(value) for value in encoded_input.values()]

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

    framework_model, _ = paddle_trace(model, inputs=inputs)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
