"""This module calculate the groundtruth values for the metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Tuple

    import torch


def calculate_loo_groundtruth(target_func: Callable, # noqa: ARG001
                              retrain_dir: str, # noqa: ARG001
                              test_dataloader: torch.utils.data.DataLoader, # noqa: ARG001
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the groundtruth values for the Leave-One-Out (LOO) metric.

    The LOO groundtruth is directly calculated by calculate the target value difference
    for each sample in the test dataloader on each model in the retrain directory.
    The target value is calculated by the target function.

    Args:
        target_func (Callable): The target function that takes a model and a dataloader
            and returns the target value. An example of a target function like follows:
            ```python
            def target_func(model, dataloader):
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader:
                        outputs = model(inputs)
                        # Do something with the outputs, e.g., calculate the loss.
                return target_value
            ```
        retrain_dir (str): The directory containing the retrained models. It should be
            the directory saved by `retrain_loo`.
        test_dataloader (torch.utils.data.DataLoader): The dataloader where each of
            the samples is used as the test set.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors. First is the LOO
            groundtruth values for each sample in test_dataloader and each model in
            retrain_dir. The returned tensor has the shape
            (num_models, num_test_samples).
            Second is the tensor indicating the removed index. The returned tensor has
            the shape (num_models,).
    """
    return None


def calculate_lds_groundtruth(target_func: Callable, # noqa: ARG001
                              retrain_dir: str, # noqa: ARG001
                              test_dataloader: torch.utils.data.DataLoader, # noqa: ARG001
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the groundtruth values for the Linear Datamodeling Score (LDS) metric.

    The LDS groundtruth is directly calculated by calculate the target value for each
    sample in the test dataloader on each model in the retrain directory. The target
    value is calculated by the target function.

    Args:
        target_func (Callable): The target function that takes a model and a dataloader
            and returns the target value. An example of a target function like follows:
            ```python
            def target_func(model, dataloader):
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader:
                        outputs = model(inputs)
                        # Do something with the outputs, e.g., calculate the loss.
                return target_value
            ```
        retrain_dir (str): The directory containing the retrained models. It should be
            the directory saved by `retrain_subset`.
        test_dataloader (torch.utils.data.DataLoader): The dataloader where each of
            the samples is used as the test set.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors. First is the LDS
            groundtruth values for each sample in test_dataloader and each model in
            retrain_dir. The returned tensor has the shape
            (num_models, num_test_samples).
            Second is the tensor indicating the sampled index.
            The returned tensor has the shape (num_models, sampled_num).
    """
    return None
