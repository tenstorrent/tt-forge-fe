# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import pybuda
import pybuda.config

def test_torch_training():
    class MultParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.rand(1, 1024))

        def forward(self, x1):
            return torch.multiply(x1, self.p)

    model = MultParam()
    shape = (1, 1024)
    inputs = torch.rand(shape)

    # Fake targets
    target = torch.zeros(shape)

    loss_fn = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    tt_model = pybuda.compile(model, sample_inputs=[torch.rand(shape)], loss=loss_fn, optimizer=optimizer)

    num_epochs = 100

    for epoch in range(num_epochs):

        print(f"parameter value: {model.p.data}")
        golden = model(inputs)
        output = tt_model(inputs)

        if not torch.allclose(output[0], golden, rtol=1e-1):
            raise ValueError("Output does not match the golden output")

        optimizer.zero_grad()

        loss = loss_fn(output[0], target)
        loss.backward()

        print(f"epoch: {epoch} loss: {loss}")
        print(f"output.grad: {output[0].grad}")
        
        loss_grad = output[0].grad
        assert loss_grad is not None

        print(f"loss grad: {loss_grad}")
        grad = tt_model.backward(loss_grad)

        # HACK to run the optimizer step
        # i'm not sure what's the right way to tie the torch optimizer to our params,
        # but this can be done automatically after backward() (hidden from user)
        model.p.grad = grad[0]

        optimizer.step()

