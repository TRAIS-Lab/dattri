"""This module implement some utility functions for the algorithm module."""

import warnings

import torch
from torch.utils.data import RandomSampler


def _check_shuffle(dataloader: torch.data.utils.DataLoader) -> None:
    """Check if the dataloader is shuffling the data.

    Args:
        dataloader (torch.data.utils.DataLoader): The dataloader to be checked.
    """
    is_shuffling = isinstance(dataloader.sampler, RandomSampler)
    if is_shuffling:
        warnings.warn(
            "The dataloader is shuffling the data. The influence \
                        calculation could not be interpreted in order.",
            stacklevel=1,
        )
