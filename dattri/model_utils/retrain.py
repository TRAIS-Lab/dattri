"""This module contains functions that retrain model for LOO/LDS/PBRF metrics."""

# ruff: noqa: ARG001, TCH002
# TODO: Remove the above line after finishing the implementation of the functions.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
import os
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import List, Optional

def retrain_loo(train_func: Callable,
                dataloader: torch.utils.data.DataLoader,
                path: str,
                indices: Optional[List[int]] = None,
                seed: Optional[int] = None) -> None:
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

    all_indices = list(range(len(dataloader.dataset)))
    if indices is None:
        # If indices are not provided default to retrain with every data.
        indices = all_indices

    metadata = {
        "mode": "loo",
        "data_length": len(dataloader),
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
        modified_dataloader = DataLoader(dataset_subset,
                                         batch_size=dataloader.batch_size)
        # Call the user specified train_func.
        model = train_func(modified_dataloader)
        # Update the metadata.
        metadata["map_index_dir"][index] = model_dir
        torch.save(model, weights_dir)

    metadata_file = Path(path) / "metadata.yml"
    with Path(metadata_file).open("w") as file:
        yaml.dump(metadata, file)


def retrain_lds(train_func: Callable,
                dataloader: torch.utils.data.DataLoader,
                path: str,
                subset_number: int = 100,
                subset_ratio: float = 0.1,
                subset_average_run: int = 1,
                seed: Optional[int] = None) -> None:
    """Retrain the model for Linear Datamodeling Score (LDS) metric.

    The retrained model checkpoints and the subset index metadata are saved
    to the `path`. The function will call the `train_func` to retrain the model
    for each subset `dataloader` with a random subset of the data.

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
        subset_number (int): The number of subsets to retrain. Default is 100.
        subset_ratio (float): The ratio of the subset to the whole dataset.
            Default is 0.1.
        subset_average_run (int): The number of times to train on one subset,
            used to remove the randomness of the training process. Default is 1.
        seed (int): The random seed for the training process and subset sampling.
            Default is None, which means the training process and subset sampling
            is not deterministic.
        path (str): The directory to save the retrained models and the subset
            index metadata. The directory should be organized as
            ```
                /$path
                    metadata.yml
                    /0
                        model_weights.pt
                    ...
                    /N
                        model_weights.pt

                # metadata.yml
                data = {
                    'mode': 'lds',
                    'data_length': len(dataloader),
                    'train_func': train_func.__name__,
                    'subset_number': subset_number,
                    'subset_ratio': subset_ratio,
                    'subset_average_run': subset_average_run,
                    'map_subset_dir': {
                        0: './0',
                        ...
                    }
                }
            ```

    Returns:
        None
    """
    path = Path(path)

    # initialize random seed and create directory
    if seed is not None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
    if not path.exists():
        path.mkdir(parents=True)

    total_data_length = len(dataloader)
    subset_length = int(total_data_length * subset_ratio)

    # Create metadata to save
    metadata = {
        "mode": "lds",
        "data_length": total_data_length,
        "train_func": train_func.__name__,
        "subset_number": subset_number,
        "subset_ratio": subset_ratio,
        "subset_average_run": subset_average_run,
        "map_subset_dir": {},
    }

    # Retrain the model for each subset
    for i in range(subset_number):
        rng = np.random.default_rng(seed)
        indices = rng.choice(total_data_length, subset_length, replace=False)
        subset_dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(dataloader.dataset, indices),
            batch_size=dataloader.batch_size,
            shuffle=True,
        )

        for _ in range(subset_average_run):
            model = train_func(subset_dataloader)
            model_path = path / str(i) / "model_weights.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)

        metadata["map_subset_dir"][i] = str(path / str(i))

    # Save metadata
    with (path / "metadata.yml").open("w") as f:
        yaml.dump(metadata, f)
