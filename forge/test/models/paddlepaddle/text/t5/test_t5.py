# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle
from paddlenlp.transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer, T5Model

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["t5-small",
    "t5-base",
    # "t5-large",
    ]

@pytest.mark.xfail() # Fail in tracing - to be fixed
@pytest.mark.parametrize("variant", variants)
def test_t5_encoder(forge_property_recorder, variant):
    model = T5ForConditionalGeneration.from_pretrained(variant)
    tokenizer = T5Tokenizer.from_pretrained(variant)

    encoder = model.t5.encoder

    input = "Write a poem about the sea"
    encoded_input = tokenizer(input)

    inputs = [paddle.to_tensor([encoded_input["input_ids"]])]

    # Test framework model
    outputs = encoder(*inputs)
    print(outputs)
    
    # Trace the model   
    class WrappedT5Encoder(paddle.nn.Layer):
        def __init__(self, model):
            super(WrappedT5Encoder, self).__init__()
            self.model = model
        def forward(self, input_ids):
            return self.model(input_ids=input_ids)
    model = WrappedT5Encoder(encoder)

    outputs = model(*inputs)
    print(outputs)

    framework_model,_ = paddle_trace(model, inputs=inputs)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs)
    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_t5_conditional_generation(forge_property_recorder, variant):
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="t5",
        variant=variant[3:],
        task=Task.TEXT_GENERATION,
        source=Source.PADDLENLP,
    )

    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    model = T5ForConditionalGeneration.from_pretrained(variant)
    tokenizer = T5Tokenizer.from_pretrained(variant)

    inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = [paddle.to_tensor([v]) for v in inputs.values()]

    # Test framework model
    output = model(*inputs, labels=inputs[0])
    logits = output[1]

    decoded_output = paddle.argmax(logits, axis=-1)
    decoded_text = tokenizer.decode(decoded_output.numpy()[0], skip_special_tokens=True)
    print("Generated text:", decoded_text)

    # Trace the model
    class WrappedT5Model(paddle.nn.Layer):
        def __init__(self, model):
            super(WrappedT5Model, self).__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    model = WrappedT5Model(model)
    framework_model,_ = paddle_trace(model, inputs=inputs)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
    

@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_t5_model(forge_property_recorder, variant):
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="t5",
        variant=variant[3:],
        task=Task.TEXT_GENERATION,
        source=Source.PADDLENLP,
    )

    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    model = T5Model.from_pretrained(variant)
    tokenizer = T5Tokenizer.from_pretrained(variant)

    inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    input_ids = paddle.to_tensor([inputs["input_ids"]])
    decoder_input_ids = input_ids
    inputs = [input_ids, decoder_input_ids]

    # Test framework model
    outputs = model(input_ids = input_ids, decoder_input_ids=decoder_input_ids)
    print(outputs)

    # Trace the model
    class WrappedT5Model(paddle.nn.Layer):
        def __init__(self, model):
            super(WrappedT5Model, self).__init__()
            self.model = model
        def forward(self, input_ids, decoder_input_ids):
            return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    model = WrappedT5Model(model)

    framework_model,_ = paddle_trace(model, inputs=inputs)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)