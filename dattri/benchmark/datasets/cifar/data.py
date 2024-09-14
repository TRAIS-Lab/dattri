"""This module contains functions for creating the CIFAR-2 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from typing import Tuple

    from torchvision.datasets.cifar import CIFAR10

from torch.utils.data import Dataset, Subset


def get_cifar2_indices_and_adjust_labels(dataset: CIFAR10) -> List:
    """Get the CIFAR-2 indices and re-label.

    Args:
        dataset (CIFAR10): The full CIFAR-10 dataset.

    Returns:
        List of indices within the desired classes.
    """
    # Selecting 'cat' (class 3) and 'dog' (class 5) and
    # adjusting labels to 0 and 1
    indices = []
    cat_class = 3
    dog_class = 5
    for i in range(len(dataset)):
        if dataset.targets[i] == cat_class:
            indices.append(i)
            dataset.targets[i] = 0
        elif dataset.targets[i] == dog_class:
            indices.append(i)
            dataset.targets[i] = 1
    return indices


def create_cifar2_dataset(
    path: str,
) -> Tuple[Dataset, Dataset]:
    """Create CIFAR-2 dataset.

    Args:
        path (str): Root directory of the CIFAR-2 Dataset. If the
            dataset is not yet downloaded, this function will download
            it automatically to this path.

    Returns:
        Tuple[Dataset, Dataset]: The training and test CIFAR-2
            datasets.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
    )

    full_train_dataset = datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=transform,
    )
    full_test_dataset = datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=transform,
    )

    # get indices correspond to cat and dog
    # and re-label the class cat and dog
    cifar2_indices = get_cifar2_indices_and_adjust_labels(full_train_dataset)
    cifar2_indices_test = get_cifar2_indices_and_adjust_labels(full_test_dataset)

    # get the subset of the original CIFAR-10 dataset
    train_dataset = Subset(full_train_dataset, cifar2_indices)
    test_dataset = Subset(full_test_dataset, cifar2_indices_test)

    # give "data" and "targets" method
    train_dataset.data = full_train_dataset.data[cifar2_indices]
    train_dataset.targets = [full_train_dataset.targets[idx] for idx in cifar2_indices]
    test_dataset.data = full_test_dataset.data[cifar2_indices_test]
    test_dataset.targets = [
        full_test_dataset.targets[idx] for idx in cifar2_indices_test
    ]

    return train_dataset, test_dataset


def create_cifar_dataset(
    path: str,
) -> Tuple[Dataset, Dataset]:
    """Create CIFAR-10 dataset.

    Args:
        path (str): Root directory of the CIFAR-10 Dataset. If the
            dataset is not yet downloaded, this function will download
            it automatically to this path.

    Returns:
        Tuple[Dataset, Dataset]: The training and test CIFAR-10
            datasets.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
    )

    train_dataset = datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset
