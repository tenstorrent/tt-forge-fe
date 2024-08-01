# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

import pybuda
from .utils import *

def test_mnist_training():
    torch.manual_seed(0)

    # Config
    num_epochs = 9
    batch_size = 64
    learning_rate = 0.005
    
    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Load TensorBoard writer (for logging)
    writer = load_tb_writer()
    
    # Define model and instruct it to compile and run on TT device
    framework_model = MNISTLinear()
    tt_model = pybuda.compile(framework_model)
    tt_model.to("tt")

    # Create a torch loss and leave on CPU
    loss = torch.nn.L1Loss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    tt_optimizer = pybuda.compile(framework_optimizer)
    tt_optimizer.to("tt")

    for epoch_idx in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Put inputs on device
            data = data.to("tt")
            
            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Reset gradients (every batch)
            tt_optimizer.zero_grad()
            
            # Forward pass (prediction) on device
            pred = tt_model(data)
            
            # Pull output back to CPU
            pred = pred.to("cpu")

            # Compute loss on CPU
            loss = tt_loss(pred, target)
            
            # RUn backward pass on device
            loss.backward()
            
            # Adjust weights (on device)
            tt_optimizer.step()

            # Log gradients
            for name, param in tt_model.named_parameters():
                writer.add_histogram(f"{name}.grad", param.grad, batch_idx)

            # Log loss
            writer.add_scalar("Loss", loss.item(), batch_idx)
