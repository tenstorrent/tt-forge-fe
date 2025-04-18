# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle
from paddlenlp.transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer, T5Model

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace


variants = [
    "t5-small",
    "t5-base",
]


@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_t5_encoder(forge_property_recorder, variant):

    # Load Model and Tokenizer
    model = T5ForConditionalGeneration.from_pretrained(variant)
    tokenizer = T5Tokenizer.from_pretrained(variant)
    encoder = model.t5.encoder

    # Load sample
    input = "Write a poem about the sea"
    encoded_input = tokenizer(input)
    inputs = [paddle.to_tensor([encoded_input["input_ids"]]), paddle.to_tensor([encoded_input["attention_mask"]])]

    # Test framework model
    outputs = encoder(*inputs)
    print(outputs)

    # Trace the model
    class WrappedT5Encoder(paddle.nn.Layer):
        def __init__(self, model):
            super(WrappedT5Encoder, self).__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
            )

    model = WrappedT5Encoder(encoder)

    # Test framework model
    outputs = model(*inputs)
    print(outputs)

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
