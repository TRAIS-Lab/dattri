"""This module contains functions for model training/evaluation on the MNIST dataset."""
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


def train_mnist_lr(dataloader: torch.utils.data.DataLoader) -> LogisticRegressionMnist:
    """Train a logistic regression model on the MNIST dataset.

    Args:
        dataloader: The dataloader for the MNIST dataset.

    Returns:
        The trained logistic regression model.
    """
    model = LogisticRegressionMnist()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model


def loss_mnist_lr(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    """Calculate the loss of the logistic regression model on the MNIST dataset.

    Args:
        model: The logistic regression model.
        dataloader: The dataloader for the MNIST dataset.

    Returns:
        The sum of loss of the model on the loader.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]
    return total_loss / total_samples
