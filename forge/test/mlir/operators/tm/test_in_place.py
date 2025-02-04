# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import tensorflow as tf

import forge
from forge.verify.compare import compare_with_golden


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
    ],
)
@pytest.mark.push
def test_in_place_torch(shape):
    class Inplace(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = x + 1
            x += 2

            return x + y

    input = torch.zeros(shape, requires_grad=False)
    framework_input = input.detach().clone()
    tt_inputs = [input]

    framework_model = Inplace()
    y = framework_model(framework_input)

    compiled_model = forge.compile(framework_model, sample_inputs=tt_inputs, module_name="inplace")
    tty = compiled_model(*tt_inputs)[0]

    compare_with_golden(golden=y, calculated=tty)
    print(framework_input)
    print(tt_inputs[0])


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
    ],
)
@pytest.mark.push
def test_in_place_tf(shape):
    class Inplace(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x):
            y = x + 1
            x += 2
            return x + y

    input = tf.zeros(shape)
    framework_input = tf.identity(input)
    tt_inputs = [input]

    framework_model = Inplace()
    y = framework_model(framework_input)

    compiled_model = forge.compile(framework_model, sample_inputs=tt_inputs, module_name="inplace")
    tty = compiled_model(*tt_inputs)[0]

    # convert tensor from tf to torch
    y = torch.tensor(y.numpy())

    compare_with_golden(golden=y, calculated=tty)

    print(framework_input)
    print(tt_inputs[0])


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1024),
    ],
)
@pytest.mark.push
@pytest.mark.xfail(
    reason="TT model silently computes the result for the tensor with requires_grad=True on which in-place operation has been performed"
)
def test_in_place_backward(shape):
    class MatmulParam(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.rand(1024, 1024))
            torch.nn.init.xavier_uniform_(self.p)

        def forward(self, x):
            y = torch.matmul(x, self.p)
            x += 2  # In-place modification causing runtime error
            return y

    inputs = torch.rand(shape, requires_grad=True)

    model = MatmulParam()
    tt_model = forge.compile(model, sample_inputs=[torch.rand(shape)])

    # Expect PyTorch model to raise a RuntimeError
    with pytest.raises(
        RuntimeError, match="a leaf Variable that requires grad is being used in an in-place operation."
    ):
        model(inputs)

    # Expect tt_model to also raise a RuntimeError
    with pytest.raises(RuntimeError):
        tt_model(inputs)  # If no error is raised, the test will fail
