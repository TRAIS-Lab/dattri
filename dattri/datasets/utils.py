"""This module contains some utils functions to process datasets."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, Union

import numpy as np
import torch


def _random_flip(label: any, label_space: set, seed: int = 42) -> any:
    """Helper function for flip_label.

    The function performs a random selection of label from the label space.

    Args:
        label (any): The label tensor to be flipped.
        label_space (set): The valid range of labels given in a set
        seed (int): Random seed.

    Returns:
        any: The randomly selected label to replace the original one
    """
    label_space.discard(label)
    rng = np.random.default_rng(seed)
    target_label = rng.choice(list(label_space))
    label_space.add(label)
    return target_label


def flip_label(label: Union[np.ndarray, torch.Tensor],
               label_space: Union[list, np.ndarray, torch.Tensor] = None,
               p: float = 0.1,
               seed: int = 42) -> Tuple[Union[np.ndarray, torch.Tensor], list]:
    """Flip the label of the input label tensor with the probability `p`.

    The function will randomly select a new label from the `label_space` to replace
    the original label.

    Args:
        label (Union[np.ndarray, torch.Tensor]): The label tensor to be flipped.
        label_space (Union[list, np.ndarray, torch.Tensor]): The label space to
            sample the new label. If None, the label space will be inferred from the
            unique values in the input label tensor.
        p (float): The probability to flip the label.
        seed (int): Random seed.

    Returns:
        Tuple[Union[np.ndarray, torch.Tensor], list]: A tuple of two elements.
            The first element is the flipped label tensor. The second element is
            the flipped indices.
    """
    if p <= 0.0 or p >= 1:
        message = "Noise ratio must be a float number between 0 and 1"
        raise ValueError(message)

    if label_space is None:
        label_space = np.unique(label)

    label_space = set(label_space)

    n_train = len(label)
    rng = np.random.default_rng(seed)
    noise_index = rng.choice(n_train,
                            size=int(p * n_train),
                            replace=False)

    # Deep copy to avoid in-place modification
    flipped_label = copy.deepcopy(label)

    # Generate a list of randomly sampled noisy (flipped) data from label space
    noisy_data = np.vectorize(
        lambda x: _random_flip(x, label_space, seed),
    )(flipped_label[noise_index])

    if isinstance(flipped_label, torch.Tensor):
        noisy_data = torch.tensor(noisy_data)

    # Flip the labels
    flipped_label[noise_index] = noisy_data

    return flipped_label, noise_index
