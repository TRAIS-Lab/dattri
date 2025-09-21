"""This module contains functions for model training/evaluation on the ImageNet."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

logger = logging.getLogger(__name__)


def train_imagenet_resnet18(
    dataloader: torch.utils.data.DataLoader,
    seed: int = 0,
    device: str = "cpu",
    num_epochs: int = 15,
) -> resnet18:
    """Train a ResNet18 on the ImageNet dataset.

    Args:
        dataloader: The dataloader for the ImageNet dataset.
        seed: The seed for training the model.
        device: The device to train the model on.
        num_epochs: The number of training epoch.

    Returns:
        (nn.Module): The trained resnet18 model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    model = create_resnet18_model()
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.8,
        weight_decay=4e-5,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs_t, labels_t = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model(inputs_t)
            loss = criterion(outputs, labels_t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs_t.size(0)
            message = (
                f"Epoch {epoch + 1}/{num_epochs}, Step: {i}, Step Loss: {loss.item()}"
            )
            logger.info(message)
        epoch_loss = running_loss / len(dataloader.dataset)
        message = f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}"
        logger.info(message)

    return model


def loss_imagenet_resnet18(
    model_path: str,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> float:
    """Calculate the loss of the ResNet18 on the ImageNet dataset.

    Args:
        model_path: The path to the saved model weights.
        dataloader: The dataloader for the ImageNet dataset.
        device: The device to evaluate the model on.

    Returns:
        float: The sum of loss of the model on the loader.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model = create_resnet18_model()
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


def create_resnet18_model() -> resnet18:
    """Create a ResNet18 model.

    Returns:
        The ResNet18 model.
    """
    return resnet18(pretrained=False)
