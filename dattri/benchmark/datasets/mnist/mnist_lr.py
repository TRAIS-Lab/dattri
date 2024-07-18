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
    epoch_num: int = 20,
) -> LogisticRegressionMnist:
    """Train a logistic regression model on the MNIST dataset.

    Args:
        dataloader: The dataloader for the MNIST dataset.
        seed: The seed for training the model.
        device: The device to train the model on.
        epoch_num: The number of epochs to train the model.

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
        float: The per-example loss of the model on the loader.
    """
    criterion = nn.CrossEntropyLoss(reduction="none")
    model = create_lr_model("mnist")
    model.load_state_dict(torch.load(Path(model_path)))
    model.eval()
    model.to(device)
    loss_list = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss_list.append(loss.clone().detach().cpu())
    return torch.cat(loss_list)
