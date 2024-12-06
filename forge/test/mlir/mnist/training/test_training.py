# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time

import torch
from torch import nn
from loguru import logger

import forge
from forge.op.loss import CrossEntropyLoss, L1Loss
from ..utils import *
from forge.op.eval.common import compare_with_golden
from forge.verify.config import VerifyConfig
from forge.tensor import to_forge_tensors


@pytest.mark.push
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
    framework_model = MNISTLinear(bias=False)  # bias=False because batch_size=1 with bias=True is not supported

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    tt_model = forge.compile(framework_model, sample_inputs=[torch.rand(batch_size, 784)], training=True)

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
            assert compare_with_golden(golden_pred, pred, verify_cfg=VerifyConfig(pcc=0.95))

            # Compute loss on CPU
            loss = loss_fn(pred, target)
            total_loss += loss.item()

            golden_loss = loss_fn(golden_pred, target)
            assert torch.allclose(loss, golden_loss, rtol=5e-2)  # 5% tolerance

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


@pytest.mark.parametrize("freeze_layer", [None, 0, 2, 4])
@pytest.mark.push
def test_forge_vs_torch_gradients(freeze_layer):
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

    if freeze_layer is not None:
        forge_model.linear_relu_stack[freeze_layer].weight.requires_grad = False
        forge_model.linear_relu_stack[freeze_layer].bias.requires_grad = False
        torch_model.linear_relu_stack[freeze_layer].weight.requires_grad = False
        torch_model.linear_relu_stack[freeze_layer].bias.requires_grad = False

    loss_fn = nn.CrossEntropyLoss()

    sample_inputs = [torch.ones(batch_size, in_features, dtype=dtype)]

    tt_model = forge.compile(forge_model, sample_inputs=sample_inputs, training=True)

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

    if freeze_layer is not None:
        assert forge_model.linear_relu_stack[freeze_layer].weight.grad is None
        assert forge_model.linear_relu_stack[freeze_layer].bias.grad is None

    # Compare gradients for each parameter
    for name in reversed(list(torch_grads.keys())):
        assert compare_with_golden(torch_grads[name], forge_grads[name])


# For bfloat16, the following line should be added to the test_forge_vs_torch function:
# In file forge/forge/op/eval/forge/eltwise_unary.py:418 should be replaced with: threshold_tensor = ac.tensor(torch.zeros(shape, dtype=torch.bfloat16) + threshold)
# That sets relu threshold to bfloat16 tensor.
# And in file forge/forge/compile.py::compile_main forced bfloat 16 should be added compiler_cfg.default_df_override = DataFormat.Float16_b
# @pytest.mark.skip(reason="Need to be tested with bfloat16 and takes around 10 minutes to run")
@pytest.mark.push
def test_forge_vs_torch():
    torch.manual_seed(0)

    batch_size = 64
    learning_rate = 1e-2
    epochs = 10
    verbose = True

    dtype = torch.bfloat16

    torch_model = MNISTLinear(dtype=dtype, bias=False)
    forge_model = MNISTLinear(dtype=dtype, bias=False)

    copy_params(torch_model, forge_model)

    torch_writer = load_tb_writer("torch")
    forge_writer = load_tb_writer("forge")

    loss_fn = nn.CrossEntropyLoss()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
    forge_optimizer = forge.optimizers.SGD(learning_rate=learning_rate)

    tt_model = forge.compile(
        forge_model, sample_inputs=[torch.rand(batch_size, 784, dtype=dtype)], optimizer=forge_optimizer, training=True
    )

    test_loader, train_loader = load_dataset(batch_size, dtype=dtype)
    step = 0

    loss_inputs = [torch.rand(batch_size, 10, dtype=dtype).requires_grad_(True), torch.rand(batch_size, 10)]
    loss_inputs = to_forge_tensors(loss_inputs)
    forge_loss_fn = CrossEntropyLoss(name="cross_entropy_loss")
    tt_loss = forge.compile(forge_loss_fn, sample_inputs=loss_inputs, attach_to=tt_model, training=True)


    early_stop = EarlyStopping(patience=10, mode="max")

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
            is_tt=False,
            verbose=verbose,
        )
        forge_loop = train_loop(
            train_loader,
            tt_model,
            tt_loss,
            forge_optimizer,
            batch_size,
            forge_model.named_parameters,
            is_tt=True,
            verbose=verbose,
        )
        for torch_data, forge_data in zip(torch_loop, forge_loop):
            step += 1

            torch_loss, torch_pred, torch_grads = torch_data
            forge_loss, forge_pred, forge_grads = forge_data

            if step % 100 == 0:
                torch_val_loss, torch_val_acc = validation_loop(
                    test_loader, torch_model, loss_fn, batch_size, is_tt=False
                )
                forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, is_tt=True)

                torch_writer.add_scalar("train_loss", torch_loss, step)
                forge_writer.add_scalar("train_loss", forge_loss, step)
                torch_writer.add_scalar("validation_acc", torch_val_acc, step)
                forge_writer.add_scalar("validation_acc", forge_val_acc, step)

                torch_writer.flush()
                forge_writer.flush()

        if verbose:
            print(f"Epoch {i} took {time.time() - start_time} seconds")

        forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, is_tt=True)
        early_stop.step(forge_val_acc, i)

        # if early_stop.is_best():
        #     torch.save(torch_model.state_dict(), f"runs/models/torch_model_{i}.pth")
        #     torch.save(forge_model.state_dict(), f"runs/models/forge_model_{i}.pth")

        if early_stop.is_early_stop():
            break

    # # Load best model
    # torch_model.load_state_dict(torch.load(f"runs/models/torch_model_{early_stop.get_best_model()}.pth"))
    # forge_model.load_state_dict(torch.load(f"runs/models/forge_model_{early_stop.get_best_model()}.pth"))

    # torch_val_loss, torch_val_acc = validation_loop(test_loader, torch_model, loss_fn, batch_size, is_tt=False)
    # forge_val_loss, forge_val_acc = validation_loop(test_loader, tt_model, loss_fn, batch_size, is_tt=True)

    # print(f"Validation accuracy for Torch: {torch_val_acc} in epoch {early_stop.get_best_model()}")
    # print(f"Validation accuracy for Forge: {forge_val_acc} in epoch {early_stop.get_best_model()}")


@pytest.mark.push
def test_loss_device():
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
    writer = load_tb_writer("forge_mnist")

    framework_model = MNISTLinear(bias=False)
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)

    tt_model = forge.compile(framework_model, sample_inputs=[torch.rand(batch_size, 784)], training=True)

    loss_fn = CrossEntropyLoss(name="cross_entropy_loss")

    loss_inputs = [torch.rand(batch_size, 10).requires_grad_(True), torch.rand(batch_size, 10)]
    loss_inputs = to_forge_tensors(loss_inputs)

    tt_loss = forge.compile(loss_fn, sample_inputs=loss_inputs, attach_to=tt_model, training=True)

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
            print(f"pred: {pred}")
            print(f"golden_pred: {golden_pred}")
            assert compare_with_golden(golden_pred, pred, VerifyConfig(pcc=0.95))

            loss = tt_loss(pred, target)
            total_loss += loss[0].item()

            # Run backward pass on device
            tt_loss.backward()

            if batch_idx >= limit_num_batches:
                break

        print(f"epoch: {epoch_idx} loss: {total_loss}")

        # Adjust weights (on CPU)
        tt_model.print_gradients()

        for name, param in framework_model.named_parameters():
            if param.grad is not None:
                print(f"param: {name} grad: {param.grad}")

        framework_optimizer.step()

    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = tt_model(data)[0]
        target = nn.functional.one_hot(target, num_classes=10).float()

        test_loss += tt_loss(pred, target)[0]

        if batch_idx == 1000:
            break

    print(f"Test (total) loss: {test_loss}")


def test_golden_mnist():
    torch.manual_seed(0)

    # Config
    num_epochs = 10
    batch_size = 2048
    learning_rate = 0.1

    # Limit number of batches to run - quicker test
    limit_num_batches = 1

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Load TensorBoard writer (for logging)
    writer = load_tb_writer("forge_mnist")

    framework_model = MNISTLinear(bias=False)
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    framework_loss = torch.nn.CrossEntropyLoss()

    losses = []
    for epoch_idx in range(num_epochs):
        # Reset gradients (every epoch) - since our batch size is currently 1,
        # we accumulate gradients across multiple batches (limit_num_batches),
        # and then run the optimizer.
        framework_optimizer.zero_grad()

        print(f"====================== Epoch {epoch_idx} ======================")
        print("Framework model params:")
        for name, param in framework_model.named_parameters():
            print(f"param: {name} value: {param.data}")

        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            golden_pred = framework_model(data)
            golden_loss = framework_loss(golden_pred, target)
            golden_loss /= limit_num_batches
            total_loss += golden_loss.item()

            golden_loss.backward()

            if batch_idx >= limit_num_batches:
                print(f"target: {target}")
                print(f"golden.shape: {golden_loss.shape}")
                break

        print(f"epoch: {epoch_idx} loss: {total_loss}")
        losses.append(total_loss)

        print(f"Framework model gradients:")
        for name, param in framework_model.named_parameters():
            print(f"param: {name} grad: {param.grad}")
        print(f"=============================================================")
        framework_optimizer.step()

    # test_loss = 0
    # for batch_idx, (data, target) in enumerate(test_loader):
    #     pred = tt_model(data)[0]
    #     target = nn.functional.one_hot(target, num_classes=10).float()
    #
    #     test_loss += tt_loss(pred, target)[0]
    #
    #     if batch_idx == 1000:
    #         break
    #
    # print(f"Test (total) loss: {test_loss}")
    print(f"Losses: {losses}")


@pytest.mark.push
def test_e2e_device():
    torch.manual_seed(0)

    # Config
    num_epochs = 1000
    batch_size = 1024
    learning_rate = 0.1

    # Limit number of batches to run - quicker test
    limit_num_batches = 1

    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Load TensorBoard writer (for logging)
    writer = load_tb_writer("forge_mnist")

    framework_model = MNISTLinear(bias=False)
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    optimizer = forge.optimizers.SGD(learning_rate=learning_rate)

    tt_model = forge.compile(
        framework_model, sample_inputs=[torch.rand(batch_size, 784)], optimizer=optimizer, training=True
    )

    loss_fn = CrossEntropyLoss(name="cross_entropy_loss")

    loss_inputs = [torch.rand(batch_size, 10).requires_grad_(True), torch.rand(batch_size, 10)]
    loss_inputs = to_forge_tensors(loss_inputs)

    tt_loss = forge.compile(loss_fn, sample_inputs=loss_inputs, attach_to=tt_model, training=True)

    logger.info("Starting training loop... (logger will be disabled)")
    logger.disable("")

    losses = []
    for epoch_idx in range(num_epochs):
        # Reset gradients (every epoch) - since our batch size is currently 1,
        # we accumulate gradients across multiple batches (limit_num_batches),
        # and then run the optimizer.
        framework_optimizer.zero_grad()

        print(f"====================== Epoch {epoch_idx} ======================")
        print("Framework model params:")
        for name, param in framework_model.named_parameters():
            print(f"param: {name} value: {param.data}")
        print(f"=============================================================")

        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            pred = tt_model(data)[0]
            golden_pred = framework_model(data)
            assert compare_with_golden(golden_pred, pred, VerifyConfig(pcc=0.95))

            loss = tt_loss(pred, target)
            total_loss += loss[0].item()

            # Run backward pass on device
            tt_loss.backward()

            if batch_idx >= limit_num_batches:
                break

        print(f"epoch: {epoch_idx} loss: {total_loss}")
        if total_loss > 10:
            total_loss = 10
        losses.append(total_loss)

        tt_model.print_gradients()
        optimizer.step()
        tt_model.print_gradients()

    # test_loss = 0
    # for batch_idx, (data, target) in enumerate(test_loader):
    #     pred = tt_model(data)[0]
    #     target = nn.functional.one_hot(target, num_classes=10).float()
    #
    #     test_loss += tt_loss(pred, target)[0]
    #
    #     if batch_idx == 1000:
    #         break
    #
    # print(f"Test (total) loss: {test_loss}")
    print(f"Losses: {losses}")
