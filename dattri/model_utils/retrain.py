"""This module contains functions that retrain model for LOO/LDS/PBRF metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import List, Optional

    import torch


def retrain_loo(train_func: Callable, # noqa: ARG001
                dataloader: torch.utils.data.DataLoader, # noqa: ARG001
                path: str, # noqa: ARG001
                indices: Optional[List[int]] = None, # noqa: ARG001
                seed: Optional[int] = None) -> None: # noqa: ARG001
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

    Returns:
        None
    """
    return

def retrain_lds(train_func: Callable, # noqa: ARG001, PLR0913
                dataloader: torch.utils.data.DataLoader, # noqa: ARG001
                path: str, # noqa: ARG001
                subset_number: int = 100, # noqa: ARG001
                subset_ratio: float = 0.1, # noqa: ARG001
                subset_average_run: int = 1, # noqa: ARG001
                seed: Optional[int] = None) -> None: # noqa: ARG001
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
        (None) None
    """
    return
