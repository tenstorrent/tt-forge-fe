# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
import operator

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST as mnist_dataset


# Model definition
class MNISTLinear(nn.Module):
    def __init__(
        self, input_size=784, output_size=10, hidden_size=512, bias=True, dtype=torch.float32
    ):  # changed hidden_size to 512 because matmul 256 x batch_size is not supported in ttnn
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


class LoraLayer(nn.Module):
    def __init__(self, input_size, output_size, rank=8, alpha=4, dtype=torch.float32):
        super(LoraLayer, self).__init__()
        self.a = nn.Linear(in_features=input_size, out_features=rank, bias=False, dtype=dtype)
        self.b = nn.Linear(in_features=rank, out_features=output_size, bias=False, dtype=dtype)
        self.alpha = alpha / rank

        nn.init.kaiming_uniform_(self.a.weight, a=torch.sqrt(torch.tensor([5])).item())
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        logits = self.a(x)
        logits = self.alpha * self.b(logits)

        return logits


class MNISTLora(nn.Module):
    def __init__(
        self, input_size=784, output_size=10, hidden_size=512, bias=True, rank=8, alpha=16, dtype=torch.float32
    ):
        super(MNISTLora, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=bias, dtype=dtype)
        self.lora1 = LoraLayer(input_size, hidden_size, rank=rank, alpha=alpha, dtype=dtype)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.lora2 = LoraLayer(hidden_size, hidden_size, rank=rank, alpha=alpha, dtype=dtype)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(hidden_size, output_size, bias=bias, dtype=dtype)

        self.freeze_linear_layers()

    def forward(self, x):
        first_layer_logits = self.relu1(self.linear1(x) + self.lora1(x))
        second_layer_logits = self.relu2(self.linear2(first_layer_logits) + self.lora2(first_layer_logits))
        final_logits = self.linear3(second_layer_logits)

        return final_logits

    def freeze_linear_layers(self):
        for layer in [self.linear1, self.linear2, self.linear3]:
        # for layer in [self.linear1, self.linear3]:
            for param in layer.parameters():
                param.requires_grad = False


class EarlyStopping:
    def __init__(self, patience=3, mode="max"):
        assert mode in ["min", "max"]
        if mode == "min":
            self.better = operator.lt
        elif mode == "max":
            self.better = operator.gt
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_current_best = False
        self.best_model = None

    def step(self, val_metric, model_id):
        if self.best_score is None or self.better(val_metric, self.best_score):
            self.is_current_best = True
            self.best_score = val_metric
            self.counter = 0
            self.best_model = model_id
        else:
            self.is_current_best = False
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True

    def is_best(self):
        return self.is_current_best

    def is_early_stop(self):
        return self.early_stop

    def get_best_model(self):
        return self.best_model


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
    return {name: param.grad.detach().clone() for name, param in named_params() if param.grad is not None}


def copy_params(src, dst):
    state_dict = src.state_dict()
    for name, param in dst.named_parameters():
        param.data = state_dict[name].data.detach().clone()

    dst.load_state_dict(state_dict)


def write_grads(writer, named_params, step):
    for name in named_params:
        writer.add_histogram(name, named_params[name].flatten().float(), step)


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, named_params, is_tt=False, verbose=False):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        pred = pred[0] if is_tt else pred

        y = nn.functional.one_hot(y, num_classes=10).to(pred.dtype)
        loss = loss_fn(pred, y)

        loss.backward()
        if is_tt:
            model.backward()

        yield loss, pred, get_param_grads(named_params)

        optimizer.step()
        optimizer.zero_grad()

        if verbose and batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"{'Forge' if is_tt else 'Torch'} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation_loop(dataloader, model, loss_fn, batch_size, is_tt=False, verbose=False):
    size = len(dataloader.dataset)
    loss, accuracy = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred = pred[0] if is_tt else pred
            y = nn.functional.one_hot(y, num_classes=10).to(pred.dtype)
            loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    loss /= size
    accuracy /= size
    if verbose:
        print(
            f"{'Forge' if is_tt else 'Torch'} Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n"
        )
    return loss, accuracy
