"""This module contains some utils function to process datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, Union

    import numpy as np
    import torch


def flip_label(label: Union[np.ndarray, torch.Tensor], # noqa: ARG001
               label_space: Union[list, np.ndarray, torch.Tensor] = None, # noqa: ARG001
               p: float = 0.1) -> Tuple[Union[np.ndarray, torch.Tensor], list]: # noqa: ARG001
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
    return None
