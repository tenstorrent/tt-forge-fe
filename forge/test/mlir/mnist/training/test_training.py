# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

import forge
from ..utils import *
from forge.op.eval.common import compare_with_golden

from loguru import logger

def test_mnist_training():
    torch.manual_seed(0)

    # Config
    num_epochs = 3
    batch_size = 1
    learning_rate = 0.001

    # Limit number of batches to run - quicker test
    limit_num_batches = 1000
    
    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Load TensorBoard writer (for logging)
    writer = load_tb_writer()
    
    # Define model and instruct it to compile and run on TT device
    framework_model = MNISTLinear()

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    tt_model = forge.compile(framework_model, sample_inputs=[torch.rand(batch_size, 784)], loss=loss_fn, optimizer=framework_optimizer)

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")
    for epoch_idx in range(num_epochs):
        # Reset gradients (every epoch) - since our batch size is currently 1,
        # we accumulate gradients across multiple batches (limit_num_batches),
        # and then run the optimizer.
        framework_optimizer.zero_grad()

        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            pred = tt_model(data)[0]
            golden_pred = framework_model(data)
            compare_with_golden(golden_pred, pred)

            # Compute loss on CPU
            loss = loss_fn(pred, target)
            total_loss += loss.item()

            golden_loss = loss_fn(golden_pred, target)
            compare_with_golden(golden_loss, loss)
            
            # Run backward pass on device
            loss.backward()
            
            tt_model.backward(pred.grad)

            if batch_idx >= limit_num_batches:
                break

        print(f"epoch: {epoch_idx} loss: {total_loss}")

        # Adjust weights (on CPU)
        framework_optimizer.step()

    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = tt_model(data)[0]
        target = nn.functional.one_hot(target, num_classes=10).float()

        test_loss += loss_fn(pred, target)

        if batch_idx == 1000:
            break

    print(f"Test (total) loss: {test_loss}")
