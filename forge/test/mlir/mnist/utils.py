# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST as mnist_dataset


# Model definition
class MNISTLinear(nn.Module):
    def __init__(self, input_size=784, output_size=10, hidden_size=512, bias=True, dtype=torch.float32):
        super(MNISTLinear, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=bias, dtype=dtype),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        is_current_best = False
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            is_current_best = True
            self.best_score = val_metric
            self.counter = 0

        return is_current_best


def load_tb_writer(model):
    """
    Load TensorBoard writer for logging
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/gradient_visualization/{model}/{current_time}/"
    writer = SummaryWriter(log_dir)

    return writer


def load_dataset(batch_size, dtype=torch.float32):
    """
    Load and normalize MNIST dataset
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std for MNIST
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten image
            transforms.Lambda(lambda x: x.to(dtype)),  # Convert to dtype
        ]
    )

    train_dataset = mnist_dataset(root="./data", train=True, download=True, transform=transform)
    test_dataset = mnist_dataset(root="./data", train=False, download=True, transform=transform)

    # Shuffle training data so that shuffling is not done in the training loop
    # This is to ensure that the same data is used for both Torch and Forge
    indices = torch.randperm(len(train_dataset))
    train_dataset = [train_dataset[i] for i in indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return test_loader, train_loader


def get_param_grads(named_params):
    return {name: param.grad.detach().clone() for name, param in named_params()}


def copy_params(src, dst):
    state_dict = src.state_dict()
    for name, param in dst.named_parameters():
        param.data = state_dict[name].data.detach().clone()

    dst.load_state_dict(state_dict)


def write_grads(writer, named_params, step):
    for name in named_params:
        writer.add_histogram(name, named_params[name].flatten().float(), step)


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, named_params, isTT=False, verbose=False):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        pred = pred[0] if isTT else pred

        y = nn.functional.one_hot(y.long(), num_classes=10).to(pred.dtype)
        loss = loss_fn(pred, y)

        if isTT:
            loss.backward()
            model.backward(pred.grad)
        else:
            loss.backward()

        yield loss, pred, get_param_grads(named_params)

        optimizer.step()
        optimizer.zero_grad()

        if verbose and batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"{'Forge' if isTT else 'Torch'} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation_loop(dataloader, model, loss_fn, batch_size, isTT=False, verbose=False):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred = pred[0] if isTT else pred
            y = nn.functional.one_hot(y.long(), num_classes=10).to(pred.dtype)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    if verbose:
        print(
            f"{'Forge' if isTT else 'Torch'} Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
    return test_loss, correct
