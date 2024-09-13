"""This module provides brittleness prediction to evaluate the data attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import List, Optional

    from torch.utils.data import DataLoader

import numpy as np
import torch
from tqdm import tqdm

from dattri.benchmark.utils import SubsetSampler


def brittleness(
    train_loader: DataLoader,
    test_loader: DataLoader,
    scores: torch.Tensor,
    train_func: Callable[[DataLoader], torch.nn.Module],
    eval_func: Callable[[torch.nn.Module, DataLoader], torch.Tensor],
    device: torch.device = "cpu",
    search_space: Optional[List[int]] = None,
) -> Optional[int]:
    """Calculate smallest k to make a test data prediction flip.

    This function calculate the brittleness metric by determining the smallest subset of
    training data whose removal causes the test sample's prediction to flip.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for test data.
        scores (torch.Tensor):
            Attribution scores in shape (len(train_loader), len(test_loader)).
        train_func (Callable[[DataLoader], torch.nn.Module]):
            Function to retrain the model on modified dataset.
        eval_func (Callable[[torch.nn.Module, DataLoader], torch.Tensor]):
            Function to evaluate the model and return predictions.
        device (torch.device): Computation device (CPU or GPU).
        search_space (List[int]):
            List of points in the search space for most influential training data.

    Returns:
        int or None: The smallest number of influential training points whose removal
        flips the test point's prediction, or None if no such k exists.
    """
    if search_space is None:
        search_space = list(range(0, 200, 20))
    k_values = list(search_space)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy() if scores.is_cuda else scores.numpy()
    sorted_indices = np.argsort(scores)[::-1]
    for k in tqdm(
        k_values,
        desc="Calculating brittleness",
        leave=False,
    ):
        highest_score_indices = sorted_indices[:k]
        if check_if_flip(
            train_loader,
            test_loader,
            highest_score_indices,
            train_func,
            eval_func,
            device,
        ):
            return k
    return None


def check_if_flip(
    train_loader: DataLoader,
    test_loader: DataLoader,
    indices_to_remove: List[int],
    train_func: Callable[[DataLoader], torch.nn.Module],
    eval_func: Callable[[torch.nn.Module, DataLoader], torch.Tensor],
    device: torch.device = "cpu",
) -> bool:
    """Check if a test sample flips after removing specified training data.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for test data.
        indices_to_remove (List[int]): Indices of training data to remove.
        train_func (Callable[[DataLoader], torch.nn.Module]):
            Function to retrain the model on modified dataset.
        eval_func (Callable[[torch.nn.Module, DataLoader], torch.Tensor]):
            Function to evaluate the model and return predictions.
        device (torch.device): Computation device.

    Returns:
        bool: True if prediction flips after data removal, otherwise False.
    """
    original_sampler = train_loader.sampler
    original_indices = original_sampler.indices
    remaining_indices = list(set(original_indices) - set(indices_to_remove))

    new_train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=SubsetSampler(remaining_indices),
    )

    model = train_func(new_train_loader)
    model.to(device)
    model.eval()

    for test_data in test_loader:
        x, label = test_data
        x, label = x.to(device), label.to(device)
    pred = eval_func(model, test_loader)
    return pred.item() != label.item()
