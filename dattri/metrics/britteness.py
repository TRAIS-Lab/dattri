"""This module provides britteness prediction evaluate the data attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def brittleness(
    test_data: Tuple[torch.Tensor, torch.Tensor],
    train_loader: DataLoader,
    train_data_indices: torch.Tensor,
    scores: torch.Tensor,
    steplength: Tuple[int, int, int],
    train_func: Callable[[DataLoader], torch.nn.Module],
    device: torch.device,
    batch_size: int = 64,
) -> Optional[int]:
    """Calculate smallest k make a test data flip.

    This function calculate the brittleness metric by determining the smallest subset of
    training data whose removal causes the test sample's prediction to flip.

    Args:
        test_data (tuple): A tuple containing the image and label of the test sample.
        train_loader (DataLoader): DataLoader for the training data.
        train_data_indices (list[int]): Indices of the training data used.
        scores (torch.Tensor):
            Data attribution scores associated with each training sample.
        steplength (tuple): Tuple containing start, end, and step size for the k range.
        train_func (function): Function to retrain the model on the modified dataset.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        batch_size (int):
            The batch size used when creating new DataLoaders. Default is 64.

    Returns:
        int or None: The smallest number of influential training points whose removal
        flips the test point's prediction, or None if no such k exists.
    """
    start, end, step = steplength
    k_values = list(range(start, end, step))
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy() if scores.is_cuda else scores.numpy()
    sorted_indices = np.argsort(scores)[::-1]
    for k in tqdm(
        k_values,
        desc="Calculating brittleness",
        leave=False,
        ):
        highest_score_indices = sorted_indices[:k]
        if check_if_flip(test_data,
                         train_loader,
                         train_data_indices,
                         highest_score_indices,
                         train_func,
                         device,
                         batch_size):
            return k
    return None


def check_if_flip(
    test_data: Tuple[torch.Tensor, torch.Tensor],
    train_loader: DataLoader,
    train_data_indices: List[int],
    indices_to_remove: List[int],
    train_func: Callable[[DataLoader], torch.nn.Module],
    device: torch.device,
    batch_size: int,
) -> bool:
    """Flip check for brittleness prediction.

    This function weill check if the prediction for a test sample flips after
    removing certain training data.

    Args:
        test_data (tuple): The test data consisting of an image and its label.
        train_loader (DataLoader): DataLoader for the training data.
        train_data_indices (list[int]):
            List of indices indicating orignial data the used training data.
        indices_to_remove (list[int]):
            Indices of the training data to remove.
        train_func (function): Function to train the model on the modified dataset.
        device (torch.device): The device for computation.
        batch_size (int): Batch size for the DataLoader of the new training set.

    Returns:
        bool: True if the prediction flips, otherwise False.
    """
    dataset = train_loader.dataset
    dataset = Subset(dataset, train_data_indices)

    remaining_indices = list(set(range(len(dataset))) - set(indices_to_remove))
    subset = Subset(dataset, remaining_indices)
    new_train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    model = train_func(new_train_loader)
    model.to(device)
    model.eval()

    x, label = test_data
    x, label = x.to(device), label.to(device)

    output = model(x)
    pred = output.argmax(dim=1, keepdim=True)

    return pred.item() != label.item()
