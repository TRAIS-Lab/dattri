"""This module contains functions for model training/evaluation on the MNIST dataset."""

from pathlib import Path

import torch
from torch import nn


class LogisticRegressionMnist(nn.Module):
    """A simple logistic regression model for MNIST dataset."""

    def __init__(self) -> None:
        """Initialize the logistic regression model."""
        super(LogisticRegressionMnist, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the logistic regression model.

        Args:
            x: The input image tensor.

        Returns:
            The output tensor.
        """
        x = x.view(x.shape[0], -1)  # Flatten the image
        return self.linear(x)


def train_mnist_lr(
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> LogisticRegressionMnist:
    """Train a logistic regression model on the MNIST dataset.

    Args:
        dataloader: The dataloader for the MNIST dataset.
        device: The device to train the model on.

    Returns:
        The trained logistic regression model.
    """
    model = LogisticRegressionMnist()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    model.to(device)
    epoch_num = 20
    for _ in range(epoch_num):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

    return model


def loss_mnist_lr(
    model_path: str,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> float:
    """Calculate the loss of the logistic regression model on the MNIST dataset.

    Args:
        model_path: The path to the saved model weights.
        dataloader: The dataloader for the MNIST dataset.
        device: The device to evaluate the model on.

    Returns:
        The sum of loss of the model on the loader.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model = LogisticRegressionMnist()
    model.load_state_dict(torch.load(Path(model_path)))
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]
    return total_loss / total_samples
