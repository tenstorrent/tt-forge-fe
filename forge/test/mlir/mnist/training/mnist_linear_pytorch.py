# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

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

def train():
    torch.manual_seed(777)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
        ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    writer = SummaryWriter()

    num_epochs = 10
    input_size = 784
    hidden_size = 256
    output_size = 10
    model = FeedForward(input_size, hidden_size, output_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    best_accuracy = 0.0
    best_checkpoint = None

    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        accuracy = 100.0 * total_correct / total_samples
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy
            }

    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint['model_state_dict'])
        optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
        print(f'Reverted to checkpoint with highest validation accuracy: {best_checkpoint["accuracy"]:.2f}%')

    writer.close()

if __name__ == "__main__":
    train()
