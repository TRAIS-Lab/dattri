"""This module contains functions for LR training/evaluation on the MNIST dataset."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dattri.benchmark.models.mlp import MLPMnist, create_mlp_model


def train_mnist_mlp(
    dataloader: torch.utils.data.DataLoader,
    seed: int = 0,
    device: str = "cpu",
) -> MLPMnist:
    """Train a MLP model on the MNIST dataset.

    Args:
        dataloader: The dataloader for the MNIST dataset.
        seed: The seed for training the model.
        device: The device to train the model on.

    Returns:
        The trained MLP model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    model = create_mlp_model("mnist")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    model.to(device)
    epoch_num = 50
    for _ in range(epoch_num):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

    return model


def loss_mnist_mlp(
    model_path: str,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> float:
    """Calculate the loss of the MLP model on the MNIST dataset.

    Args:
        model_path: The path to the saved model weights.
        dataloader: The dataloader for the MNIST dataset.
        device: The device to evaluate the model on.

    Returns:
        The sum of loss of the model on the loader.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model = create_mlp_model("mnist")
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
