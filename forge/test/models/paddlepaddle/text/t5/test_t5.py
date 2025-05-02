# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle
from paddlenlp.transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer, T5Model

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from forge.forge_property_utils import Framework, Source, Task

variants = [
    "t5-small",
    "t5-base",
]


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_t5_conditional_generation(forge_property_recorder, variant):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="t5",
        variant=variant[3:],
        task=Task.TEXT_GENERATION,
        source=Source.PADDLENLP,
    )
    forge_property_recorder.record_group("generality")

    # Load Model and Tokenizer
    model = T5ForConditionalGeneration.from_pretrained(variant)
    tokenizer = T5Tokenizer.from_pretrained(variant)

    # Load sample
    inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = [paddle.to_tensor([v]) for v in inputs.values()]

    # Test framework model
    output = model(*inputs, labels=inputs[0])
    logits = output[1]
    decoded_output = paddle.argmax(logits, axis=-1)
    decoded_text = tokenizer.decode(decoded_output.numpy()[0], skip_special_tokens=True)
    print("Generated text:", decoded_text)

    # Wrap Model to fix input signature
    class WrappedT5Model(paddle.nn.Layer):
        def __init__(self, model):
            super(WrappedT5Model, self).__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

    model = WrappedT5Model(model)

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        framework_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
