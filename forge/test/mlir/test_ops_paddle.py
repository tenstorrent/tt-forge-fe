# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from forge.tvm_calls.forge_utils import paddle_trace
import paddle
import pytest
import torch

import forge
from forge.verify.verify import verify


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

    # After setting input_spec in compile function, the framework_model is changed and can no longer have torch tensor inputs.
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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_matmul_pp():
    class Matmul_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return paddle.matmul(a, b)

    inputs = [paddle.rand([32, 64]), paddle.rand([64, 32])]

    framework_model = Matmul_pp()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_squeeze_pp():
    class Squeeze_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            squeezed_a = paddle.squeeze(a, axis=0)
            squeezed_b = paddle.squeeze(b, axis=0)
            return squeezed_a.T + squeezed_b

    inputs = [paddle.rand([1, 32, 32]), paddle.rand([1, 32, 32])]

    framework_model = Squeeze_pp()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_multiple_layers_pp():
    class CNNClassifier_pp(paddle.nn.Layer):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
            self.conv2 = paddle.nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.fc1 = paddle.nn.Linear(in_features=32 * 8 * 8, out_features=128)
            self.fc2 = paddle.nn.Linear(in_features=128, out_features=num_classes)

        def forward(self, x):
            x = self.pool(paddle.nn.functional.relu(self.conv1(x)))
            x = self.pool(paddle.nn.functional.relu(self.conv2(x)))
            x = paddle.flatten(x, start_axis=1)
            x = paddle.nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    inputs = [paddle.rand([1, 3, 32, 32])]

    framework_model = CNNClassifier_pp()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_mnist_linear_pp():
    class PaddleMNISTLinear(paddle.nn.Layer):
        def __init__(self, input_size=784, output_size=10, hidden_size=512, bias=True):
            super(PaddleMNISTLinear, self).__init__()
            self.model = paddle.nn.Sequential(
                paddle.nn.Linear(input_size, hidden_size, bias_attr=bias),
                paddle.nn.ReLU(),
                paddle.nn.Linear(hidden_size, hidden_size, bias_attr=bias),
                paddle.nn.ReLU(),
                paddle.nn.Linear(hidden_size, output_size, bias_attr=bias),
            )

        def forward(self, x):
            logits = self.model(x)
            return logits

    inputs = [paddle.rand([1, 784])]

    framework_model = PaddleMNISTLinear()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_loaded_model():
    input_features, output_dim = (784, 10)

    class Linear_pp(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.l1 = paddle.nn.Linear(input_features, output_dim, bias_attr=True)

        def forward(self, a):
            return self.l1(a)

    inputs = [paddle.rand([1, input_features])]

    input_spec = [paddle.static.InputSpec(shape=[1, input_features], dtype="float32")]
    framework_model, _ = paddle_trace(Linear_pp(), input_spec)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_batchnorm_pp():
    class BatchNorm_pp(paddle.nn.Layer):
        def __init__(self, num_features):
            super().__init__()
            self.batch_norm = paddle.nn.BatchNorm2D(num_features)

        def forward(self, x):
            return self.batch_norm(x)

    inputs = [paddle.rand((1, 32, 56, 56))]

    framework_model = BatchNorm_pp(num_features=32)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
def test_convbn_pp():
    class ConvBNLayer(paddle.nn.Layer):
        def __init__(self, in_c, out_c, filter_size, stride, padding, num_groups=1, if_act=True, act=None):
            super(ConvBNLayer, self).__init__()
            self.if_act = if_act
            self.act = act

            self.conv = paddle.nn.Conv2D(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                groups=num_groups,
            )

            self.bn = paddle.nn.BatchNorm(num_channels=out_c)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            if self.if_act:
                if self.act == "relu":
                    x = paddle.nn.functional.relu(x)
                elif self.act == "hardswish":
                    x = paddle.nn.functional.hardswish(x)
                else:
                    print("The activation function is selected incorrectly.")
                    exit()
            return x

    inputs = [paddle.randn([1, 3, 64, 64])]

    framework_model = ConvBNLayer(in_c=3, out_c=64, filter_size=3, stride=1, padding=1, if_act=True, act="relu")
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
