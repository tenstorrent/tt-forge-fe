# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import forge
from forge import (
    CPUDevice,
    PyTorchModule,
)
from utils import (
    MNISTLinear,
    Identity,
    load_tb_writer,
    load_dataset,
)
from forge.config import _get_global_compiler_config


class FeedForward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(loss_on_cpu=True):
    torch.manual_seed(777)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x: x.view(-1))]
    )
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    writer = SummaryWriter()

    num_epochs = 2
    input_size = 784
    hidden_size = 256
    output_size = 10
    batch_size = 3
    learning_rate = 0.001
    sequential = True

    framework_model = FeedForward(input_size, hidden_size, output_size)
    tt_model = forge.PyTorchModule(f"mnist_linear_{batch_size}", framework_model)
    tt_optimizer = forge.optimizers.SGD(learning_rate=learning_rate, device_params=True)
    tt0 = forge.TTDevice("tt0", module=tt_model, optimizer=tt_optimizer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Dataset sample input
    first_sample = test_loader.dataset[0]
    sample_input = (first_sample[0].repeat(1, batch_size, 1),)
    sample_target = (
        torch.nn.functional.one_hot(torch.tensor(first_sample[1]), num_classes=output_size)
        .float()
        .repeat(1, batch_size, 1)
    )

    if loss_on_cpu:
        cpu0 = CPUDevice("cpu0", module=PyTorchModule("identity", Identity()))
        cpu0.place_loss_module(forge.PyTorchModule(f"loss_{batch_size}", torch.nn.CrossEntropyLoss()))
    else:
        tt_loss = forge.PyTorchModule(f"loss_{batch_size}", torch.nn.CrossEntropyLoss())
        tt0.place_loss_module(tt_loss)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_auto_fusing = False

    if not loss_on_cpu:
        sample_target = (sample_target,)

    checkpoint_queue = forge.initialize_pipeline(
        training=True,
        sample_inputs=sample_input,
        sample_targets=sample_target,
        _sequential=sequential,
    )

    best_accuracy = 0.0
    best_checkpoint = None

    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):

            images = (images.unsqueeze(0),)
            tt0.push_to_inputs(images)

            targets = torch.nn.functional.one_hot(labels, num_classes=output_size).float().unsqueeze(0)
            if loss_on_cpu:
                cpu0.push_to_target_inputs(targets)
            else:
                tt0.push_to_target_inputs(targets)

            forge.run_forward(input_count=1, _sequential=sequential)
            forge.run_backward(input_count=1, zero_grad=True, _sequential=sequential)
            forge.run_optimizer(checkpoint=True, _sequential=sequential)

            loss_q = forge.run.get_loss_queue()

            step = 0
            loss = loss_q.get()[0]
            print(loss)
            # while not loss_q.empty():
            #     if loss_on_cpu:
            #         writer.add_scalar("Loss/Forge/overfit", loss_q.get()[0], step)
            #     else:
            #         writer.add_scalar("Loss/Forge/overfit", loss_q.get()[0].value()[0], step)
            #     step += 1

    writer.close()


if __name__ == "__main__":
    train()
