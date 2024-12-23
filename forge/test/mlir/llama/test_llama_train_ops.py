# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F
import pytest

import forge
from forge.verify.verify import verify
from forge.op.loss import CrossEntropyLoss
from forge.tensor import to_forge_tensors


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param(
            (1, 12, 200),
            -1,
        ),
        pytest.param(
            (128, 12, 200),
            -1,
        ),
        pytest.param(
            (256, 12, 200),
            -1,
        ),
        pytest.param(
            (1, 12, 200),
            -2,
        ),
        pytest.param(
            (128, 12, 200),
            -2,
        ),
        pytest.param(
            (256, 12, 200),
            -2,
        ),
    ],
)
@pytest.mark.push
@pytest.mark.xfail(
    reason="RuntimeError: Failed to run MLIR compiler pass pipeline. error: 'ttnn.reshape' op Shape attribute size must match output tensor rank. Tracking on: https://github.com/tenstorrent/tt-mlir/issues/1577"
)
def test_mean_bwd(input_shape, dim):
    class MeanBwd(nn.Module):
        def __init__(self, dim: int):
            super(MeanBwd, self).__init__()
            self.fc1 = nn.Linear(200, 3200)
            self.dim = dim

        def forward(self, x):
            return torch.mean(F.relu(self.fc1(x)), dim=self.dim)

    input_ids = torch.randn([*input_shape])

    framework_model = MeanBwd(dim=dim)
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=0.001)

    compiled_model = forge.compile(framework_model, input_ids, optimizer=framework_optimizer, training=True)

    verify([input_ids], framework_model, compiled_model)


@pytest.mark.parametrize(
        "in_features, out_features", [pytest.param(3200, 3200), pytest.param(3200, 8640), pytest.param(8640, 3200)]
)
@pytest.mark.push
def test_matmul_dims(in_features, out_features):
    class MatMulDimsCheck(nn.Module):
        def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
            super(MatMulDimsCheck, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(784, in_features, bias=bias, dtype=dtype),
                nn.ReLU(),
                nn.Linear(in_features, out_features, bias=bias, dtype=dtype),
                nn.ReLU(),
                nn.Linear(out_features, 10, bias=bias, dtype=dtype),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    torch.manual_seed(0)

    framework_model = MatMulDimsCheck(in_features=in_features, out_features=out_features, bias=False)
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=0.001)
    tt_model = forge.compile(framework_model, sample_inputs=[torch.rand(12, 784)], training=True)

    loss_fn = CrossEntropyLoss(name="cross_entropy_loss")
    loss_inputs = [torch.rand(12, 10).requires_grad_(True), torch.rand(12, 10)]
    loss_inputs = to_forge_tensors(loss_inputs)
    tt_loss = forge.compile(loss_fn, sample_inputs=loss_inputs, attach_to=tt_model, training=True)

    framework_optimizer.zero_grad()

    # Create target tensor and leave on CPU
    target = torch.nn.functional.one_hot(torch.randint(0, 9, (12,)), num_classes=10).float()

    # Forward pass (prediction) on device
    input_ids = torch.randn((12, 784))
    pred = tt_model(input_ids)[0]

    tt_loss(pred, target)

    # Run backward pass on device
    tt_loss.backward()

    # Adjust weights (on CPU)
    framework_optimizer.step()
