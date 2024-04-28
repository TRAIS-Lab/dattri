"""This module contains functions for LR training/evaluation on the MNIST dataset."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dattri.benchmark.models.logistic_regression import (
    LogisticRegressionMnist,
    create_lr_model,
)


def train_mnist_lr(
    dataloader: torch.utils.data.DataLoader,
    seed: int = 0,
    device: str = "cpu",
) -> LogisticRegressionMnist:
    """Train a logistic regression model on the MNIST dataset.

    Args:
        dataloader: The dataloader for the MNIST dataset.
        seed: The seed for training the model.
        device: The device to train the model on.

    Returns:
        The trained logistic regression model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    model = create_lr_model("mnist")
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
    model = create_lr_model("mnist")
    model.load_state_dict(torch.load(Path(model_path)))
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]
    return total_loss / total_samples


def create_mnist_dataset(
    path: str,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create MNIST dataset for training and testing.

    Returns:
        Tuple[torchvision.dataset, torchvision.dataset]: The training dataset,
            and the testing dataset.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ],
    )
    dataset_train = datasets.MNIST(path, train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST(path, train=False, download=True, transform=transform)

    return dataset_train, dataset_test
