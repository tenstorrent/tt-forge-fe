# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import paddle

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from paddlenlp.transformers import ErnieForSequenceClassification, ErnieForMaskedLM, ErnieTokenizer

def test_multi_head_attention():
    model = paddle.nn.MultiHeadAttention(
        embed_dim=128,
        num_heads=2
    )

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
    encoder = model.ernie.encoder
    
    hidden_size = model.config.hidden_size
    input = paddle.rand((1, 12, hidden_size))
    inputs = [input]

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

def test_ernie_model():
    model_name = "ernie-1.0"
    model = ErnieForSequenceClassification.from_pretrained(model_name, num_classes=2)
    model = model.ernie
    input = paddle.randint(0, 100, (1, 12))
    inputs = [input]
    compiled_model = forge.compile(model, inputs)
    verify(inputs, model, compiled_model)

def test_ernie_for_sequence_classification(forge_property_recorder):
    model_name = "ernie-1.0"
    model = ErnieForSequenceClassification.from_pretrained(model_name, num_classes=2)
    input = paddle.randint(0, 100, (1, 12))
    inputs = [input]
    compiled_model = forge.compile(model, inputs)
    verify(inputs, model, compiled_model, forge_property_recorder=forge_property_recorder)

