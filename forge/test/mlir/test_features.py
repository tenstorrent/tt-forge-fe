# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.push
@pytest.mark.functional
def test_multiple_inputs(forge_property_recorder):
    class MultipleInputs(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b, c):
            return a + b + c

    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32), torch.rand(1, 32, 32)]

    framework_model = MultipleInputs()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    [
        ((1, 1, 32, 64), (1, 1, 64, 128), (1, 1, 128, 32)),
    ],
)
@pytest.mark.push
@pytest.mark.functional
def test_input_order(forge_property_recorder, a_shape, b_shape, c_shape):
    class InputOrderWithConstants(nn.Module):
        def __init__(self):
            super().__init__()
            self.const1 = torch.rand(1, 1, 32, 32)
            self.const2 = torch.rand(1, 1, 32, 32)

        def forward(self, a, b, c):
            x = torch.matmul(a, b)
            x = torch.matmul(x, c)
            x = x + self.const1
            x = x * self.const2
            return x

    a = torch.rand(*a_shape)
    b = torch.rand(*b_shape)
    c = torch.rand(*c_shape)

    framework_model = InputOrderWithConstants()
    compiled_model = forge.compile(
        framework_model, sample_inputs=[a, b, c], forge_property_handler=forge_property_recorder
    )

    verify([a, b, c], framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("batch_size", [1, 4, 16, 32, 64])
@pytest.mark.parametrize("linear_features", [(784, 10)])
@pytest.mark.push
@pytest.mark.functional
def test_matmul_bias(forge_property_recorder, batch_size, linear_features):
    input_features, output_dim = linear_features

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(input_features, output_dim, bias=True)

        def forward(self, a):
            return self.l1(a)

    inputs = [torch.rand(batch_size, input_features)]

    framework_model = Linear()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("batch_size", [1, 2, 16, 64, 512])
@pytest.mark.parametrize("in_features", [784])
@pytest.mark.parametrize("out_features", [10])
@pytest.mark.push
@pytest.mark.functional
def test_batch_size_inference(forge_property_recorder, batch_size, in_features, out_features):
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x):
            y = self.linear(x)
            return nn.functional.softmax(y, dim=-1)

    in_data = [torch.rand(batch_size, in_features)]

    framework_model = SimpleModel()
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[torch.rand(batch_size, in_features)],
        forge_property_handler=forge_property_recorder,
    )

    verify(in_data, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize("batch_size", [1, 2, 16, 64, 512])
@pytest.mark.parametrize("in_features", [784])
@pytest.mark.parametrize("out_features", [10])
@pytest.mark.push
@pytest.mark.functional
def test_batch_size_training(forge_property_recorder, batch_size, in_features, out_features):
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x):
            y = self.linear(x)
            return nn.functional.softmax(y, dim=-1)

    in_data = torch.rand(batch_size, in_features)
    out_data = torch.randint(0, out_features, (batch_size,))
    target = nn.functional.one_hot(out_data, num_classes=out_features).float()

    model = SimpleModel()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    tt_model = forge.compile(
        model,
        sample_inputs=[torch.rand(batch_size, in_features)],
        optimizer=optimizer,
        forge_property_handler=forge_property_recorder,
    )

    optimizer.zero_grad()

    fw_out, tt_out = verify(
        inputs=[in_data], framework_model=model, compiled_model=tt_model, forge_property_handler=forge_property_recorder
    )
    golden_pred = fw_out[0]
    pred = tt_out[0]

    loss = loss_fn(pred, target)
    golden_loss = loss_fn(golden_pred, target)
    assert torch.allclose(loss, golden_loss, rtol=1e-2)  # 1e-2 is the minimum value for which the test passes

    loss.backward()
    tt_model.backward()
