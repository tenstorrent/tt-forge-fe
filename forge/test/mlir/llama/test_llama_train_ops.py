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
            (256, 7, 200),
            -1,
        ),
        pytest.param(
            (256, 31, 200),
            -1,
        ),
        pytest.param(
            (256, 89, 200),
            -1,
        ),
        pytest.param(
            (256, 117, 200),
            -1,
        ),
        pytest.param(
            (1, 12, 200),
            -2,
            marks=pytest.mark.xfail(reason="Data mismatch -> AutomaticValueChecker (compare_with_golden)"),
        ),
        pytest.param(
            (128, 12, 200),
            -2,
            marks=pytest.mark.xfail(reason="Data mismatch -> AutomaticValueChecker (compare_with_golden)"),
        ),
        pytest.param(
            (256, 12, 200),
            -2,
            marks=pytest.mark.xfail(reason="Data mismatch -> AutomaticValueChecker (compare_with_golden)"),
        ),
        pytest.param(
            (256, 7, 200),
            -2,
            marks=pytest.mark.xfail(reason="Data mismatch -> AutomaticValueChecker (compare_with_golden)"),
        ),
        pytest.param(
            (256, 31, 200),
            -2,
        ),
        pytest.param(
            (256, 89, 200),
            -2,
        ),
        pytest.param(
            (256, 117, 200),
            -2,
        ),
        pytest.param(
            (1, 200), -1, marks=pytest.mark.xfail(reason="Data mismatch -> AutomaticValueChecker (compare_with_golden)")
        ),
        pytest.param(
            (128, 200),
            -1,
        ),
        pytest.param(
            (1, 12, 64, 200),
            -1,
        ),
        pytest.param(
            (128, 12, 17, 200),
            -1,
        ),
        pytest.param(
            (1, 12, 64, 200),
            -2,
        ),
        pytest.param(
            (128, 12, 17, 200),
            -2,
            marks=pytest.mark.xfail(reason="Data mismatch -> AutomaticValueChecker (compare_with_golden)"),
        ),
        pytest.param(
            (1, 12, 64, 200),
            -3,
            marks=pytest.mark.xfail(reason="RuntimeError: TT_ASSERT (i >= 0) && (i < (int)dims_.size())"),
        ),
        pytest.param(
            (128, 12, 17, 200),
            -3,
            marks=pytest.mark.xfail(reason="RuntimeError: TT_ASSERT (i >= 0) && (i < (int)dims_.size())"),
        ),
    ],
)
@pytest.mark.push
@pytest.mark.functional
def test_mean_bwd(forge_property_recorder, input_shape, dim):
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

    compiled_model = forge.compile(
        framework_model,
        input_ids,
        optimizer=framework_optimizer,
        training=True,
        forge_property_handler=forge_property_recorder,
    )

    verify([input_ids], framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "in_features, out_features",
    [
        pytest.param(3200, 3200),
        pytest.param(3200, 8640),
        pytest.param(8640, 3200),
        pytest.param(8, 3200),
        pytest.param(16, 3200),
        pytest.param(32, 3200),
        pytest.param(64, 3200),
        pytest.param(128, 3200),
        pytest.param(256, 3200),
        pytest.param(512, 3200),
        pytest.param(32000, 3200),
    ],
)
@pytest.mark.push
@pytest.mark.functional
def test_matmul_dims(forge_property_recorder, in_features, out_features):
    class MatMulDimsCheck(nn.Module):
        def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
            super(MatMulDimsCheck, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(3200, in_features, bias=bias, dtype=dtype),
                nn.ReLU(),
                nn.Linear(in_features, out_features, bias=bias, dtype=dtype),
                nn.ReLU(),
                nn.Linear(out_features, 10, bias=bias, dtype=dtype),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    framework_model = MatMulDimsCheck(in_features=in_features, out_features=out_features, bias=False)
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=0.001)
    tt_model = forge.compile(
        framework_model,
        sample_inputs=[torch.rand(12, 3200)],
        training=True,
        forge_property_handler=forge_property_recorder,
    )

    loss_fn = CrossEntropyLoss(name="cross_entropy_loss")
    loss_inputs = [torch.rand(12, 10).requires_grad_(True), torch.rand(12, 10)]
    loss_inputs = to_forge_tensors(loss_inputs)
    tt_loss = forge.compile(
        loss_fn,
        sample_inputs=loss_inputs,
        attach_to=tt_model,
        training=True,
        forge_property_handler=forge_property_recorder,
    )

    framework_optimizer.zero_grad()

    # Create target tensor and leave on CPU
    target = torch.nn.functional.one_hot(torch.randint(0, 9, (12,)), num_classes=10).float()

    # Forward pass (prediction) on device
    input_ids = torch.randn((12, 3200))
    pred = tt_model(input_ids)[0]
    verify([input_ids], framework_model, tt_model, forge_property_handler=forge_property_recorder)

    tt_loss(pred, target)

    # Run backward pass on device
    tt_loss.backward()

    # Adjust weights (on CPU)
    framework_optimizer.step()
