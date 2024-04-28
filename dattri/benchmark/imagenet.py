"""This module contains functions for model training/evaluation on the ImageNet."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

if TYPE_CHECKING:
    from typing import Tuple

import logging

logging.basicConfig(level=logging.INFO)


def train_imagenet_resnet18(
    dataloader: torch.utils.data.DataLoader,
    seed: int = 0,
    device: str = "cpu",
) -> resnet18:
    """Train a ResNet18 on the ImageNet dataset.

    Args:
        dataloader: The dataloader for the ImageNet dataset.
        seed: The seed for training the model.
        device: The device to train the model on.

    Returns:
        The trained logistic regression model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    model = resnet18(pretrained=False)
    model.train()
    num_epochs = 15
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
            logging.info(message)
        epoch_loss = running_loss / len(dataloader.dataset)
        message = f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}"
        logging.info(message)

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
        The sum of loss of the model on the loader.
    """
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model = resnet18(pretrained=False)
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


def create_imagenet_dataset(
    path: str,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create ImageNet dataset.

    Args:
        path (str): Root directory of the ImageNet Dataset.
            It should be downloaded from http://www.image-net.org/.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: The training and
            testing ImageNet datasets.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    train_dataset = datasets.ImageNet(root=path, split="train", transform=transform)
    test_dataset = datasets.ImageNet(root=path, split="test", transform=transform)

    return train_dataset, test_dataset
