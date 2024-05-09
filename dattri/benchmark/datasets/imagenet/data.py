"""This module contains functions for model training/evaluation on the ImageNet."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple

    import torch


def create_imagenet_dataset(
    path: str,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create ImageNet dataset.

    Args:
        path (str): Root directory of the ImageNet Dataset.
            It should be downloaded from http://www.image-net.org/.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: The training and
            validate ImageNet datasets.
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
    val_dataset = datasets.ImageNet(root=path, split="val", transform=transform)

    return train_dataset, val_dataset
