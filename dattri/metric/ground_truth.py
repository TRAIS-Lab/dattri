"""This module provides helper functions to calculate ground truth values."""

# ruff: noqa: ARG001, TC002
# TODO: Remove the above line after finishing the implementation of the functions.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Tuple

from pathlib import Path

import torch
import yaml


def _dir_to_index(dir_name: str) -> int:
    """Helper function for calculate_loo_ground_truth.

    This function returns the directory index to sort directories
    by index instead of by alphabets.

    Args:
        dir_name (str): Directory name of saved checkpoints.

    Returns:
        int: Index of the directory,
            for example index_12 should return 12.
    """
    prefix_len = len("index_")
    return int(dir_name[prefix_len:])


def calculate_loo_ground_truth(
    target_func: Callable,
    retrain_dir: str,
    test_dataloader: torch.utils.data.DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the ground truth values for the Leave-One-Out (LOO) metric.

    The LOO ground truth is directly calculated by calculating the target value
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
            ground truth values for each sample in test_dataloader and each model in
            retrain_dir. The returned tensor has the shape
            (num_models, num_test_samples).
            Second is the tensor indicating the removed index. The returned tensor has
            the shape (num_models,).
    """
    # Get all model file paths.
    retrain_path = Path(retrain_dir)
    model_dirs = [
        d.name for d in retrain_path.iterdir()
        if d.is_dir() and d.name.startswith("index_")
    ]
    model_dirs_sorted = sorted(model_dirs, key=_dir_to_index)
    length_dir = len(model_dirs)
    length_test = len(test_dataloader.sampler)
    # List of all predictions.
    loo_results = torch.zeros(length_dir, length_test)
    model_indices = torch.empty(length_dir)
    for dir_cnt, model_file in enumerate(model_dirs_sorted):
        model_path = Path(retrain_dir) / model_file / "model_weights.pt"
        model = torch.load(model_path)
        # Calculate target function values.
        values = target_func(model, test_dataloader)
        loo_results[dir_cnt, :] = values
        # Find excluded data index from the saved path,
        # please refer to retrain_loo in dattri/model_util/retrain.py for details.
        index = _dir_to_index(model_file)
        model_indices[dir_cnt] = int(index)
    return loo_results, model_indices


def calculate_lds_ground_truth(
    target_func: Callable,
    retrain_dir: str,
    test_dataloader: torch.utils.data.DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the ground-truth values for the Linear Datamodeling Score (LDS) metric.

    Given a `target_func`, this function calculates the values of the `target_func` on
    each sample in `test_dataloader`, and for each model in `retrain_dir`. These values
    will be used as the ground-truth values for the LDS metric.

    Args:
        target_func (Callable): The target function that takes the path to a model
            checkpoint and the `test_dataloader`, and returns the target values for all
            test samples in this dataloader. Below is an example of a target function:
            ```python
            def target_func(ckpt_path, dataloader):
                params = torch.load(ckpt_path)
                model.load_state_dict(params)  # assuming model is defined somewhere
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader:
                        outputs = model(inputs)
                        # Do something with the outputs, e.g., calculate the loss.
                return target_values
            ```
            This function should return a tensor of shape `(num_test_samples,)`
            where each element is the target value for the corresponding sample.
        retrain_dir (str): The directory containing the retrained models. It should be
            the directory saved by `retrain_lds`. The directory is organized as
            ```
                /$path
                    metadata.yml
                    /0
                        model_weights_0.pt
                        model_weights_1.pt
                        ...
                        model_weights_M.pt
                        indices.txt
                    ...
                    /N
                        model_weights_0.pt
                        model_weights_1.pt
                        ...
                        model_weights_M.pt
                        indices.txt
            ```
            additionally, the `metadata.yml` file includes the following information:
            ```
                {
                    'num_subsets': N,
                    'num_runs_per_subset': M,
                    'subset_dir_map': {
                        0: './0',
                        ...
                    }
                    ...
                }
            ```
        test_dataloader (torch.utils.data.DataLoader): The test dataloader that will
            be used to calculate the target values.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors. The first one has
            the shape (num_subsets, num_test_samples), which contains the values of the
            target function calculated on all test samples under `num_subsets` models,
            each retrained on a subset of the training data. The second tensor has the
            shape (num_subsets, subset_size), where each row refers to the indices of
            the training samples used to retrain the model. The target value will be
            flipped to be consistent with the score calculated by the attributors.
    """
    retrain_dir = Path(retrain_dir)

    # Load metadata from the retrain directory.
    with Path.open(retrain_dir / "metadata.yml", "r") as f:
        metadata = yaml.safe_load(f)
    num_subsets = metadata["num_subsets"]
    num_runs_per_subset = metadata["num_runs_per_subset"]
    subset_dir_map = metadata["subset_dir_map"]

    # Load the indices of the training samples for each subset.
    indices = []
    for i in range(num_subsets):
        with Path.open(Path(subset_dir_map[i]) / "indices.txt", "r") as f:
            indices.append([int(idx) for idx in f.read().splitlines()])
    indices = torch.tensor(indices)

    # Calculate the target values for each test sample under each model.
    target_values = torch.zeros(num_subsets, len(test_dataloader.sampler))
    for i in range(num_subsets):
        for j in range(num_runs_per_subset):
            ckpt_path = Path(subset_dir_map[i]) / f"model_weights_{j}.pt"
            target_values[i] += target_func(ckpt_path, test_dataloader)
    target_values /= num_runs_per_subset

    # flip the target values
    target_values = -target_values

    return target_values, indices
