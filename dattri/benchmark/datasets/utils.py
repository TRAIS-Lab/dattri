"""This module contains utility functions for datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, List

from torch.utils.data import Sampler


class SubsetSampler(Sampler):
    """Samples elements from a predefined list of indices."""

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
