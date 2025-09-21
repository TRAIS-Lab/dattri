"""This module contains functions that retrain model for LOO/LDS/PBRF metrics."""

# ruff: noqa: ARG001, TC002
# TODO: Remove the above line after finishing the implementation of the functions.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import List, Optional

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


def retrain_loo(
    train_func: Callable,
    dataloader: torch.utils.data.DataLoader,
    path: str,
    indices: Optional[List[int]] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> None:
    """Retrain the model for Leave-One-Out (LOO) metric.

    The retrained model checkpoints and the removed index metadata are saved
    to the `path`. The function will call the `train_func` to retrain the model
    for each subset `dataloader` with one index removed.

    Args:
        train_func (Callable): The training function that takes a dataloader, and
            returns the retrained model. Here is an example of a training function:
            ```python
            def train_func(dataloader):
                model = Model()
                optimizer = ...
                criterion = ...
                model.train()
                for inputs, labels in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                return model
            ```
        dataloader (torch.utils.data.DataLoader): The dataloader used for training.
        indices (List[int]): The indices to remove from the dataloader. Default is None.
            None means that each index in the dataloader will be removed in turn.
        seed (int): The random seed for the training process. Default is None,
            which means the training process is not deterministic.
        **kwargs: The arguments of `train_func` in addition to dataloader.
        path (str): The directory to save the retrained models and the removed index
            metadata. The directory should be organized as
            ```
                /$path
                    metadata.yml
                    /index_{indices[0]}
                        model_weights.pt
                    /index_{indices[1]}
                        model_weights.pt
                    ...
                    /index_{indices[n]}
                        model_weights.pt

                # metadata.yml
                data = {
                    'mode': 'loo',
                    'data_length': len(dataloader),
                    'train_func': train_func.__name__,
                    'indices': indices,
                    'map_index_dir': {
                        indices[0]: f'./index_{indices[0]}',
                        indices[1]: f'./index_{indices[1]}',
                        ...
                    }
                }
            ```.
    """
    if not Path(path).exists():
        # Create the path if not exists.
        Path(path).mkdir(parents=True)
    if seed is not None:
        # Manually set the seed.
        torch.manual_seed(seed)

    if dataloader.sampler is not None:
        all_indices = list(range(len(dataloader.sampler)))
    else:
        all_indices = list(range(len(dataloader.dataset)))

    if indices is None:
        # If indices are not provided default to retrain with every data.
        indices = all_indices

    metadata = {
        "mode": "loo",
        "data_length": len(all_indices),
        "train_func": train_func.__name__,
        "indices": indices,
        "map_index_dir": {},
    }

    for index in indices:
        remaining_indices = [idx for idx in all_indices if idx != index]
        model_dir = Path(path) / f"index_{index}"
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        # Create a subset of the dataset.
        weights_dir = Path(model_dir) / "model_weights.pt"
        dataset_subset = Subset(dataloader.dataset, remaining_indices)
        # Create a new DataLoader with this subset.
        modified_dataloader = DataLoader(
            dataset_subset,
            batch_size=dataloader.batch_size,
        )
        # Call the user specified train_func.
        model = train_func(modified_dataloader, seed=seed, **kwargs)
        # Update the metadata.
        metadata["map_index_dir"][index] = model_dir
        torch.save(model, weights_dir)

    metadata_file = Path(path) / "metadata.yml"
    with Path(metadata_file).open("w", encoding="utf-8") as file:
        yaml.dump(metadata, file)


def retrain_lds(
    train_func: Callable,
    dataloader: torch.utils.data.DataLoader,
    path: str,
    num_subsets: int = 100,
    subset_ratio: float = 0.5,
    num_runs_per_subset: int = 1,
    start_id: int = 0,
    total_num_subsets: int = 0,
    seed: Optional[int] = None,
    **kwargs,
) -> None:
    """Retrain the model for the Linear Datamodeling Score (LDS) metric calculation.

    The retrained model checkpoints and the subset data indices metadata will be
    saved to `path`. The function will call the `train_func` to retrain the model
    for each subset `dataloader` with a random subset of the data.

    Args:
        train_func (Callable): The training function that takes a dataloader, and
            returns the retrained model. Here is an example of a training function:
            ```python
            def train_func(dataloader, seed=None, **kwargs):
                model = Model()
                optimizer = ...
                criterion = ...
                model.train()
                for inputs, labels in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                return model
            ```
        dataloader (torch.utils.data.DataLoader): The dataloader used for training.
        path (str): The directory to save the retrained models and the subset
            index metadata. The directory should be organized as
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
            where N is (num_subsets - 1) and M is (num_runs_per_subset - 1).
        num_subsets (int): The number of subsets to retrain. Default is 100.
        subset_ratio (float): The ratio of the subset to the whole dataset.
            Default is 0.5.
        num_runs_per_subset (int): The number of retraining runs for each subset.
            Several runs can mitigate the randomness in training. Default is 1.
        start_id (int): The starting index for the subset directory. Default is 0.
            This is useful for parallelizing the retraining process.
        total_num_subsets (int): The total number of subsets. Default is 0, which
            means the total number of subsets is equal to `num_subsets`. This is
            useful for parallelizing the retraining process.
        seed (int): The random seed for the training process and subset sampling.
            Default is None, which means the training process and subset sampling
            is not deterministic.
        **kwargs: The arguments of `train_func` in addition to dataloader.

    Raises:
        ValueError: If `total_num_subsets` is negative.
        ValueError: If `num_subsets` does not divide `total_num_subsets`.
    """
    # Check that num_subsets and total_num_subsets are valid
    if total_num_subsets < 0:
        error_message = "total_num_subsets must be non-negative"
        raise ValueError(error_message)
    if total_num_subsets % num_subsets != 0:
        error_message = "num_subsets must divide total_num_subsets"
        raise ValueError(error_message)
    if total_num_subsets == 0:
        start_id = 0  # ignore start_id if total_num_subsets is 0

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

    data_length = len(dataloader.sampler)
    subset_length = int(data_length * subset_ratio)

    subset_dir_map = {}
    rng = np.random.default_rng(seed)  # this can also handle seed=None

    # seed control
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # Retrain the model for each subset
    for i in range(start_id, start_id + num_subsets):
        # Create a random subset of the data
        if seed is not None:
            rng = np.random.default_rng(seed + i)
        indices = rng.choice(data_length, subset_length, replace=False)
        subset_dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(dataloader.dataset, indices),
            batch_size=dataloader.batch_size,
            shuffle=True,
        )

        # Save the subset indices
        indices_path = path / str(i) / "indices.txt"
        indices_path.parent.mkdir(parents=True, exist_ok=True)
        with Path.open(indices_path, "w") as f:
            f.write("\n".join(map(str, indices)))

        # Retrain the model for the subset (for multiple runs)
        for j in range(num_runs_per_subset):
            if seed is not None:
                train_seed = seed + i * num_runs_per_subset + j
            else:
                train_seed = None
            model = train_func(subset_dataloader, seed=train_seed, **kwargs)
            model_path = path / str(i) / f"model_weights_{j}.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)

        # Save the subset directory
        subset_dir_map[i] = str(path / str(i))

    # Save metadata
    if (
        total_num_subsets == 0  # noqa: PLR1714
        or (start_id + 1) * num_subsets == total_num_subsets
    ):
        # Create metadata to save
        metadata = {
            "mode": "lds",
            "data_length": data_length,
            "train_func": train_func.__name__,
            "num_subsets": (start_id + 1) * num_subsets,
            "subset_ratio": subset_ratio,
            "num_runs_per_subset": num_runs_per_subset,
            "subset_dir_map": subset_dir_map,
        }
        with (path / "metadata.yml").open("w") as f:
            yaml.safe_dump(metadata, f)
