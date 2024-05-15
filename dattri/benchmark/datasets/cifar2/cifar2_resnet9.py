"""This module contains functions for model training/evaluation on CIFAR-2 dataset."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dattri.benchmark.models.resnet9.resnet9 import ResNet9

logging.basicConfig(level=logging.INFO)


def train_cifar2_resnet9(
    dataloader: DataLoader,
    seed: int = 0,
    device: str = "cpu",
    num_epochs: int = 10,
) -> ResNet9:
    """Train a ResNet9 on the CIFAR-2 dataset.

    Args:
        dataloader (DataLoader): The dataloader for the CIFAR-2 dataset.
        seed (int): The seed for training the model.
        device (str): The device to train the model on.
        num_epochs (int): The number of training epoch.

    Returns:
        The trained resnet9 model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    model = create_resnet9_model()
    model.train()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images_t, labels_t = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images_t)
            loss = criterion(outputs, labels_t)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            message = (
                f"Epoch {epoch + 1}/{num_epochs}, Step: {i}, Step Loss: {loss.item()}"
            )
            logging.info(message)
        epoch_loss = running_loss / len(dataloader.dataset)
        message = f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}"
        logging.info(message)

    return model


def loss_cifar2_resnet9(
    model_path: str,
    dataloader: DataLoader,
    device: str = "cpu",
) -> float:
    """Calculate the loss of the ResNet9 on the CIFAR-2 dataset.

    Args:
        model_path (str): The path to the saved model weights.
        dataloader (DataLoader): The dataloader for the CIFAR-2 dataset.
        device (str): The device to evaluate the model on.

    Returns:
        float: The sum of loss of the model on the loader.
    """
    criterion = nn.CrossEntropyLoss()
    model = create_resnet9_model()
    model.load_state_dict(torch.load(Path(model_path)))
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images_t, labels_t = images.to(device), labels.to(device)
            outputs = model(images_t)
            loss = criterion(outputs, labels_t)
            total_loss += loss.item() * images_t.shape[0]
            total_samples += images_t.shape[0]
    return total_loss / total_samples


def create_resnet9_model() -> ResNet9:
    """Create a ResNet9 model.

    Returns:
        The ResNet9 model.
    """
    return ResNet9(dropout_rate=0.1)
