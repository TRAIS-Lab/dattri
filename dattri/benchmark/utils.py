"""This module contains some utils functions to process datasets."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, List, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Sampler


def _random_flip(label: int, label_space: Set[int], rng: np.random.default_rng) -> int:
    """Helper function for flip_label.

    The function performs a random selection of label from the label space.

    Args:
        label (int): The label tensor to be flipped.
        label_space (set): The valid range of labels given in a set
        rng (np.random.default_rng): Random number generator.

    Returns:
        int: The randomly selected label to replace the original one
    """
    label_space.discard(label)
    target_label = int(rng.choice(list(label_space)))
    label_space.add(label)
    return target_label


def flip_label(
    label: Union[np.ndarray, torch.Tensor],
    label_space: Union[List, np.ndarray, torch.Tensor] = None,
    p: float = 0.1,
) -> Tuple[Union[np.ndarray, torch.Tensor], List]:
    """Flip the label of the input label tensor with the probability `p`.

    The function will randomly select a new label from the `label_space` to replace
    the original label.

    Args:
        label (Union[np.ndarray, torch.Tensor]): The label tensor to be flipped.
        label_space (Union[list, np.ndarray, torch.Tensor]): The label space to
            sample the new label. If None, the label space will be inferred from the
            unique values in the input label tensor.
        p (float): The probability to flip the label.

    Returns:
        Tuple[Union[np.ndarray, torch.Tensor], list]: A tuple of two elements.
            The first element is the flipped label tensor. The second element is
            the flipped indices.
    """
    if label_space is None:
        label_space = np.unique(label)

    label_space = set(label_space)

    n_train = len(label)
    rng = np.random.default_rng()
    noise_index = rng.choice(n_train, size=int(p * n_train), replace=False)

    # Deep copy to avoid in-place modification
    flipped_label = copy.deepcopy(label)

    # Generate a list of randomly sampled noisy (flipped) data from label space
    noisy_data = np.vectorize(
        lambda x: _random_flip(x, label_space, rng),
    )(flipped_label[noise_index])

    if isinstance(flipped_label, torch.Tensor):
        noisy_data = torch.tensor(noisy_data)

    # Flip the labels
    flipped_label[noise_index] = noisy_data.long()

    return flipped_label, list(noise_index)


class SubsetSampler(Sampler):
    """Samples elements from a predefined list of indices.

    Note that for training, the built-in PyTorch
    SubsetRandomSampler should be used. This class is for
    attributting process.
    """

    def __init__(self, indices: List[int]) -> None:
        """Initialize the sampler.

        Args:
            indices (list): A list of indices to sample from.
        """
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        """Get an iterator for the sampler.

        Returns:
            An iterator for the sampler.
        """
        return iter(self.indices)

    def __len__(self) -> int:
        """Get the number of indices in the sampler.

        Returns:
            The number of indices in the sampler.
        """
        return len(self.indices)
