"""This module contains functions for creating the MNIST dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple

    import torch


def create_mnist_dataset(
    path: str,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create MNIST dataset for training and testing.

    Args:
        path: The path to store the MNIST dataset.

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
