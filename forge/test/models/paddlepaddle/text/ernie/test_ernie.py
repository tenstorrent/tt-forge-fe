# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from forge.tvm_calls.forge_utils import paddle_trace
import pytest
import paddle

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

from paddlenlp.transformers import ErnieForSequenceClassification, ErnieForMaskedLM, ErnieTokenizer, ErnieConfig


def test_multi_head_attention():
    model = paddle.nn.MultiHeadAttention(embed_dim=128, num_heads=2)

    query = paddle.rand((1, 12, 128))
    key = paddle.rand((1, 12, 128))
    value = paddle.rand((1, 12, 128))

    inputs = [query, key, value]
    compiled_model = forge.compile(model, inputs)
    verify(inputs, model, compiled_model)


def test_transformer_encoder():
    encoder_layer = paddle.nn.TransformerEncoderLayer(
        d_model=128,
        nhead=2,
        dim_feedforward=512,
    )
    model = paddle.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)
    input = paddle.rand((1, 12, 128))
    inputs = [input]
    compiled_model = forge.compile(model, inputs)
    verify(inputs, model, compiled_model)


def test_ernie_embedding():
    model_name = "ernie-1.0"
    model = ErnieForSequenceClassification.from_pretrained(model_name, num_classes=2)
    embedding = model.ernie.embeddings

    input = paddle.randint(0, 100, (1, 12))
    inputs = [input]

    compiled_model = forge.compile(embedding, inputs)
    verify(inputs, embedding, compiled_model)


def test_ernie_encoder():
    model_name = "ernie-1.0"
    model = ErnieForSequenceClassification.from_pretrained(model_name, num_classes=2)

    hidden_size = model.config.hidden_size
    input = paddle.rand((1, 12, hidden_size))
    inputs = [input]
    encoder = model.ernie.encoder()
    compiled_model = forge.compile(encoder, inputs)
    verify(inputs, encoder, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.8)))


def test_ernie_pooler():
    model_name = "ernie-1.0"
    model = ErnieForSequenceClassification.from_pretrained(model_name, num_classes=2)
    pooler = model.ernie.pooler

    hidden_size = model.config.hidden_size
    input = paddle.rand((1, 12, hidden_size))
    inputs = [input]

    compiled_model = forge.compile(pooler, inputs)
    verify(inputs, pooler, compiled_model)


def test_ernie_parts():
    model_name = "ernie-1.0"
    input = paddle.randint(0, 100, (1, 12))
    inputs = [input]

    model = ErnieForSequenceClassification.from_pretrained(model_name, num_classes=2)
    embedding = model.ernie.embeddings
    encoder = model.ernie.encoder
    pooler = model.ernie.pooler

    class Ernie(paddle.nn.Layer):
        def __init__(self, embedding, encoder, pooler):
            super(Ernie, self).__init__()
            self.embedding = embedding
            self.encoder = encoder
            self.pooler = pooler

        def forward(self, input):
            embedding_output = self.embedding(input)
            encoder_output = self.encoder(embedding_output)
            pooler_output = self.pooler(encoder_output)
            return (encoder_output, pooler_output)

    framework_model = Ernie(embedding, encoder, pooler)
    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.8)))


def test_ernie_model():
    model_name = "ernie-1.0"
    model = ErnieForSequenceClassification.from_pretrained(model_name, num_classes=2)
    model = model.ernie
    input_ids = paddle.randint(0, 100, (1, 12), dtype="int32")
    token_type_ids = paddle.randint(0, 1, (1, 12), dtype="int32")
    position_ids = paddle.randint(0, 12, (1, 12), dtype="int32")
    attention_mask = paddle.zeros((1, 1, 1, 12))
    inputs = [input_ids, token_type_ids, position_ids, attention_mask]
    compiled_model = forge.compile(model, inputs)
    verify(inputs, model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.8)))


@pytest.mark.parametrize("variant", ["ernie-1.0"])
@pytest.mark.nightly
def test_ernie_for_sequence_classification(forge_property_recorder, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="ernie",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.PADDLENLP,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load Model and Tokenizer
    model = ErnieForSequenceClassification.from_pretrained(variant, num_classes=2)
    tokenizer = ErnieTokenizer.from_pretrained(variant)

    # Load sample
    input = ["Hello, my dog is cute"]
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)

    inputs = [
        paddle.to_tensor(value) for value in encoded_input.values()
    ]  # [input_ids, token_type_ids, position_ids, attention_mask]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)
    
    # Compile Model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

@pytest.mark.parametrize("variant", ["ernie-1.0"])
@pytest.mark.nightly
def test_ernie_maskedlm(forge_property_recorder, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="ernie",
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.PADDLENLP,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load Model and Tokenizer
    model = ErnieForMaskedLM.from_pretrained(variant)
    tokenizer = ErnieTokenizer.from_pretrained(variant)

    # Load sample
    input = ["One, [MASK], three, four"]
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)

    inputs = [
        paddle.to_tensor(value) for value in encoded_input.values()
    ]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Inference
    outputs = compiled_model(*inputs)
    logits = outputs[0]
    mask_token_index = (inputs[0] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1).item()
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))


