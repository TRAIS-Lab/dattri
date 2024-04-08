"""This module provides helper functions to calculate ground truth values."""

# ruff: noqa: ARG001, TCH002
# TODO: Remove the above line after finishing the implementation of the functions.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Tuple

import torch

import os

def calculate_loo_groundtruth(target_func: Callable,
                              retrain_dir: str,
                              test_dataloader: torch.utils.data.DataLoader,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the groundtruth values for the Leave-One-Out (LOO) metric.

    The LOO groundtruth is directly calculated by calculating the target value
    difference for each sample in the test dataloader on each model in the
    retrain directory. The target value is calculated by the target function.

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
        # Get all model file paths.
    model_files = [f for f in os.listdir(retrain_dir) if f.endswith('.pth') or f.endswith(".pt")]
    N = len(model_files)
    K = len(test_dataloader.dataset)
    # List of all predictions.
    loo_results = torch.zeros(N,K)
    model_indices = torch.empty(N)
    for i,model_file in enumerate(model_files):

        model_path = os.path.join(retrain_dir,model_file)
        #print(model_path)
        model = torch.load(model_path)
        model.eval()
        #calculate target function values
        values = target_func(model, test_dataloader)

        loo_results[i,:] = values
        # Find excluded data index from the saved path, please refer to retrain_loo in dattri/model_utils/retrain.py for details
        index = model_file.split("_")[-1].replace(".pth","").replace(".pt","")

        model_indices[i] = int(index)
    return loo_results, model_indices


def calculate_lds_groundtruth(target_func: Callable,
                              retrain_dir: str,
                              test_dataloader: torch.utils.data.DataLoader,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the groundtruth values for the Linear Datamodeling Score (LDS) metric.

    The LDS groundtruth is directly calculated by calculating the target value for each
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
