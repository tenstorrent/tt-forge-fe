# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time

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

    # Define model and instruct it to compile and run on TT device
    framework_model = MNISTLinear()

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    tt_model = forge.compile(
        framework_model, sample_inputs=[torch.rand(batch_size, 784)], loss=loss_fn, optimizer=framework_optimizer
    )

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
            assert compare_with_golden(golden_pred, pred, pcc=0.95)

            # Compute loss on CPU
            loss = loss_fn(pred, target)
            total_loss += loss.item()

            golden_loss = loss_fn(golden_pred, target)
            assert torch.allclose(loss, golden_loss, rtol=1e-2)

            # Run backward pass on device
            loss.backward()

            tt_model.backward()

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


def test_forge_vs_torch_gradients():
    logger.disable("")
    torch.manual_seed(0)
    batch_size = 64

    dtype = torch.float32
    torch.set_printoptions(precision=10)

    in_features = 28 * 28
    out_features = 10

    torch_model = MNISTLinear(dtype=dtype, bias=True)

    forge_model = MNISTLinear(dtype=dtype, bias=True)

    copy_params(torch_model, forge_model)

    loss_fn = nn.CrossEntropyLoss()

    sample_inputs = [torch.ones(batch_size, in_features, dtype=dtype)]

    tt_model = forge.compile(forge_model, sample_inputs=sample_inputs, loss=loss_fn)

    X = torch.ones(batch_size, in_features, dtype=dtype)
    y = torch.zeros(batch_size, out_features, dtype=dtype)

    torch_pred = torch_model(X)
    torch_loss = loss_fn(torch_pred, y)
    torch_loss.backward()
    torch_grads = get_param_grads(torch_model.named_parameters)

    X = torch.ones(batch_size, in_features, dtype=dtype)
    y = torch.zeros(batch_size, out_features, dtype=dtype)

    forge_pred = tt_model(X)[0]
    forge_loss = loss_fn(forge_pred, y)
    forge_loss.backward()
    tt_model.backward()
    forge_grads = get_param_grads(forge_model.named_parameters)

    # Compare gradients for each parameter
    for name in reversed(list(torch_grads.keys())):
        assert compare_with_golden(torch_grads[name], forge_grads[name])


# For bfloat16, the following line should be added to the test_forge_vs_torch function:
# In file forge/forge/op/eval/forge/eltwise_unary.py:418 should be replaced with: threshold_tensor = ac.tensor(torch.zeros(shape, dtype=torch.bfloat16) + threshold)
# That sets relu threshold to bfloat16 tensor.
# And in file forge/forge/compile.py::compile_main forced bfloat 16 should be added compiler_cfg.default_df_override = DataFormat.Float16_b
@pytest.mark.skip(reason="Need to be tested with bfloat16 and takes around 10 minutes to run")
def test_forge_vs_torch():
    torch.manual_seed(0)

    batch_size = 64
    learning_rate = 1e-2
    epochs = 10
    verobse = True

    dtype = torch.float32

    torch_model = MNISTLinear(dtype=dtype)
    forge_model = MNISTLinear(dtype=dtype)

    copy_params(torch_model, forge_model)

    torch_writer = load_tb_writer("torch")
    forge_writer = load_tb_writer("forge")

    loss_fn = nn.CrossEntropyLoss()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
    forge_optimizer = torch.optim.SGD(forge_model.parameters(), lr=learning_rate)

    tt_model = forge.compile(
        forge_model, sample_inputs=[torch.ones(batch_size, 784, dtype=dtype)], loss=loss_fn, optimizer=forge_optimizer
    )

    test_loader, train_loader = load_dataset(batch_size, dtype=dtype)
    step = 0

    earlyStop = EarlyStopping(patience=1)

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")
    for i in range(epochs):
        start_time = time.time()
        torch_loop = train_loop(
            train_loader,
            torch_model,
            loss_fn,
            torch_optimizer,
            batch_size,
            torch_model.named_parameters,
            isTT=False,
            verbose=verobse,
        )
        forge_loop = train_loop(
            train_loader,
            tt_model,
            loss_fn,
            forge_optimizer,
            batch_size,
            forge_model.named_parameters,
            isTT=True,
            verbose=verobse,
        )
        for torch_data, forge_data in zip(torch_loop, forge_loop):
            step += 1

            torch_loss, torch_pred, torch_grads = torch_data
            forge_loss, forge_pred, forge_grads = forge_data

            if step % 100 == 0:
                torch_val_loss, torch_val_acc = validation_loop(
                    test_loader, torch_model, loss_fn, batch_size, isTT=False
                )
                forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, isTT=True)

                torch_writer.add_scalar("train_loss", torch_loss.float(), step)
                forge_writer.add_scalar("train_loss", forge_loss.float(), step)
                torch_writer.add_scalar("validation_acc", torch_val_acc, step)
                forge_writer.add_scalar("validation_acc", forge_val_acc, step)

                torch_writer.flush()
                forge_writer.flush()

        if verobse:
            print(f"Epoch {i} took {time.time() - start_time} seconds")

        if earlyStop(forge_val_acc):
            torch.save(torch_model.state_dict(), "runs/models/torch_model.pth")
            torch.save(forge_model, "runs/models/forge_model.pth")

        if earlyStop.early_stop:
            break
