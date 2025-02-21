# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import paddle
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig

@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 28, 28),
        (1, 64, 28, 28),
        (1, 256, 28, 28),
        (1, 128, 14, 14),
        (1, 128, 56, 56),
    ],
)
@pytest.mark.push
def test_equal_pp(shape):
    class Equal_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return paddle.equal(x, y)

    x = paddle.rand(shape)
    y = x * 2.0

    inputs = [x, y]

    framework_model = Equal_pp()
    forge.compile(framework_model, sample_inputs=inputs)


@pytest.mark.push
def test_add_pp():
    class Add_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    inputs = [torch.rand([2, 32, 32]), torch.rand([2, 32, 32])]

    framework_model = Add_pp()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # after setting input_spec in compile function, the model can no longer have torch tensor inputs
    framework_model_clean = Add_pp()

    verify(inputs, framework_model_clean, compiled_model)
    

@pytest.mark.push
def test_arithmetic_pp():
    class Arithmetic_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return paddle.sqrt(a) + paddle.exp(b)

    inputs = [paddle.rand([2, 32, 32]), paddle.rand([2, 32, 32])]

    framework_model = Arithmetic_pp()
    forge.compile(framework_model, sample_inputs=inputs)

    # verification is done in the compile function

@pytest.mark.push
def test_matmul_pp():
    class Matmul_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return paddle.matmul(a, b)

    inputs = [paddle.rand([32, 64]), paddle.rand([64, 32])]

    framework_model = Matmul_pp()
    forge.compile(framework_model, sample_inputs=inputs)


@pytest.mark.push
def test_sqeeze_pp():
    class Squeeze_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            sqeezed_a = paddle.squeeze(a, axis=0)
            sqeezed_b = paddle.squeeze(b, axis=0)
            return sqeezed_a.T + sqeezed_b

    inputs = [paddle.rand([1, 32, 32]), paddle.rand([1, 32, 32])]

    framework_model = Squeeze_pp()
    forge.compile(framework_model, sample_inputs=inputs)


@pytest.mark.push
def test_flatten_pp():
    class Flatten_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.flatten = paddle.nn.Flatten()

        def forward(self, inputs):
            y = self.flatten(inputs)
            return y

    inputs = [paddle.rand([2, 32, 32])]

    framework_model = Flatten_pp()
    forge.compile(framework_model, sample_inputs=inputs)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ([2, 32, 32], [32, 32]),
        ([2, 32, 32], [1, 32, 32]),
        ([2, 32, 32], [1, 32, 1]),
        ([2, 32, 32], [1, 1, 32]),
        ([2, 32, 32], [1, 1, 1]),
        ([2, 32, 32], [32, 1]),
    ],
)
@pytest.mark.push
def test_broadcast_pp(shape_a, shape_b):
    class Broadcast_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    inputs = [paddle.rand(shape_a), paddle.rand(shape_b)]

    framework_model = Broadcast_pp()
    forge.compile(framework_model, sample_inputs=inputs)



@pytest.mark.push
def test_linear_layer_pp():
    input_features, output_dim = (784, 10)

    class Linear_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.l1 = paddle.nn.Linear(input_features, output_dim, bias_attr=True)

        def forward(self, a):
            return self.l1(a)

    inputs = [paddle.rand([1, input_features])]

    framework_model = Linear_pp()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    

@pytest.mark.push
def test_mnist_pp():

    class PaddleMNISTLinear(paddle.nn.Layer):
        def __init__(
            self, input_size=784, output_size=10, hidden_size=512, bias=True): 
            super(PaddleMNISTLinear, self).__init__()
            self.model = paddle.nn.Sequential(
                paddle.nn.Linear(input_size, hidden_size, bias_attr=bias),
                paddle.nn.ReLU(),
                paddle.nn.Linear(hidden_size, hidden_size, bias_attr=bias),
                paddle.nn.ReLU(),
                paddle.nn.Linear(hidden_size, output_size, bias_attr=bias)
            )
            

        def forward(self, x):
            logits = self.model(x)
            return logits

    inputs = [paddle.rand([1, 784])]

    framework_model = PaddleMNISTLinear()
    forge.compile(framework_model, sample_inputs=inputs)        

@pytest.mark.xfail(reason="Loaded models (TranslatedLayer) not supported yet")
@pytest.mark.push
def test_paddleocr():
    # downloaded from PaddleOCR repo
    model_path = "../paddleocr/PaddleOCR/pretrained_models/en_PP-OCRv3_rec_infer/inference"
    framework_model = paddle.jit.load(model_path)
    inputs = [paddle.rand([1, 3, 32, 320])]
    forge.compile(framework_model, sample_inputs=inputs)


