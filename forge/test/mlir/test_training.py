# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest

import forge
import forge.config
from forge.verify.compare import compare_with_golden


class MatmulParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.rand(1024, 1024))
        nn.init.xavier_uniform_(self.p)

    def forward(self, x):
        return torch.matmul(x, self.p)


@pytest.mark.push
def test_torch_training():
    model = MatmulParam()
    shape = (1, 1024)
    inputs = torch.rand(shape)
    # Fake targets
    target = torch.zeros(shape)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    tt_model = forge.compile(model, sample_inputs=[torch.rand(shape)], optimizer=optimizer)

    num_epochs = 20

    model.train()
    for epoch in range(num_epochs):
        golden = model(inputs)
        output = tt_model(inputs)

        output = [co.to("cpu") for co in output]
        assert compare_with_golden(golden=golden, calculated=output[0])

        optimizer.zero_grad()

        loss = loss_fn(output[0], target)
        loss.backward()

        golden_loss = loss_fn(golden, target)
        print(f"epoch: {epoch} loss: {loss}")
        print(f"epoch: {epoch} golden_loss: {golden_loss}")
        print(f"output.grad: {output[0].grad}")

        loss_grad = output[0].grad
        assert loss_grad is not None
        grad = tt_model.backward()

        # HACK to run the optimizer step
        # i'm not sure what's the right way to tie the torch optimizer to our params,
        # but this can be done automatically after backward() (hidden from user)
        model.p.grad = grad[0]

        optimizer.step()


@pytest.mark.push
@pytest.mark.parametrize("optimizer", [forge.optimizers.SGD, forge.optimizers.Adam, forge.optimizers.AdamW])
def test_compile_optimizers(optimizer):
    model = MatmulParam()
    shape = (1, 1024)

    optimizer = optimizer(learning_rate=0.1)
    tt_model = forge.compile(model, sample_inputs=[torch.rand(shape)], optimizer=optimizer)
