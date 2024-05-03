"""This module implement the attributor base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import dict

    import torch

import warnings

from torch.utils.data import RandomSampler


class BaseAttributor(ABC):
    """BaseAttributor."""

    @abstractmethod
    def __init__(self, target_func: Callable, **kwargs: dict) -> None:
        """Initialize the attributor.

        Args:
            target_func (Callable): The target function to be attributed.
            **kwargs (dict): The keyword arguments for the attributor.

        Returns:
            None.
        """

    @abstractmethod
    def cache(self, full_train_dataloader: torch.data.utils.DataLoader) -> None:
        """Precompute and cache some values for efficiency.

        Args:
            full_train_dataloader (torch.data.utils.DataLoader): The dataloader for
                the training data.

        Returns:
            None.
        """

    @abstractmethod
    def attribute(
        self,
        train_dataloader: torch.data.utils.DataLoader,
        test_dataloader: torch.data.utils.DataLoader,
    ) -> torch.Tensor:
        """Attribute the influence of the training data on the test data.

        Args:
            train_dataloader (torch.data.utils.DataLoader): The dataloader for
                the training data.
            test_dataloader (torch.data.utils.DataLoader): The dataloader for
                the test data.

        Returns:
            torch.Tensor: The influence of the training data on the test data.
        """
        is_shuffling = isinstance(train_dataloader.sampler, RandomSampler) & isinstance(
            test_dataloader.sampler,
            RandomSampler,
        )
        if is_shuffling:
            warnings.warn(
                "The dataloader is shuffling the data. The influence \
                           calculation could not be interpreted in order.",
                stacklevel=1,
            )
