"""This module contains functions for model training/evaluation on CIFAR dataset."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dattri.benchmark.models.resnet9.resnet9 import ResNet9

logger = logging.getLogger(__name__)


def train_cifar_resnet9(
    dataloader: DataLoader,
    seed: int = 0,
    device: str = "cpu",
    num_classes: int = 2,
    num_epochs: int = 10,
) -> ResNet9:
    """Train a ResNet9 on the CIFAR dataset.

    Args:
        dataloader (DataLoader): The dataloader for the CIFAR dataset.
        seed (int): The seed for training the model.
        device (str): The device to train the model on.
        num_classes (int): The number of classes in the dataset.
        num_epochs (int): The number of training epoch.

    Returns:
        The trained resnet9 model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    model = create_resnet9_model(num_classes=num_classes)
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
            logger.info(message)
        epoch_loss = running_loss / len(dataloader)
        message = f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}"
        logger.info(message)

    return model


def loss_cifar_resnet9(
    model_path: str,
    dataloader: DataLoader,
    device: str = "cpu",
    num_classes: int = 2,
) -> float:
    """Calculate the loss of the ResNet9 on the CIFAR dataset.

    Args:
        model_path (str): The path to the saved model weights.
        dataloader (DataLoader): The dataloader for the CIFAR dataset.
        device (str): The device to evaluate the model on.
        num_classes (int): The number of classes in the dataset.

    Returns:
        float: The per-example loss of the model on the loader.
    """
    criterion = nn.CrossEntropyLoss(reduction="none")
    model = create_resnet9_model(num_classes=num_classes)
    model.load_state_dict(torch.load(Path(model_path)))
    model.eval()
    model.to(device)
    loss_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images_t, labels_t = images.to(device), labels.to(device)
            outputs = model(images_t)
            loss = criterion(outputs, labels_t)
            loss_list.append(loss.clone().detach().cpu())
    return torch.cat(loss_list)


def create_resnet9_model(num_classes: int = 2) -> ResNet9:
    """Create a ResNet9 model.

    Args:
        num_classes (int): The number of classes in the dataset.

    Returns:
        The ResNet9 model.
    """
    return ResNet9(dropout_rate=0.0, num_classes=num_classes)
