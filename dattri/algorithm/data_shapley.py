"""This module implement data shapley valuation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import List, Optional, Tuple

    from dattri.task import AttributionTask

import torch
from tqdm import tqdm

from .base import BaseAttributor
from .utils import _check_shuffle


def default_dist_func(
    batch_x: Tuple[torch.Tensor],
    batch_y: Tuple[torch.Tensor],
) -> torch.Tensor:
    """Default distance function for KNN.

    Args:
        batch_x (Tuple[torch.Tensor]): The batch to be calculated
            distances on. The embeddings is default to be the
            first element.
        batch_y (Tuple[torch.Tensor]): The batch to be calculated
            distances on. The embeddings is default to be the
            first element.

    Returns:
        Tensor: The calculated Euclidean distance.
    """
    coord1 = batch_x[0]
    coord2 = batch_y[0]
    return torch.cdist(coord1, coord2)


class KNNShapleyAttributor(BaseAttributor):
    """KNN Data Shapley Attributor."""

    def __init__(
        self,
        k_neighbors: int,
        task: AttributionTask = None,
        distance_func: Optional[Callable] = None,
    ) -> None:
        """Initialize the AttributionTask.

        KNN Data Shapley Valuation is generally dataset-specific.
        Passing a model is optional and currently can be done in the
        customizable distance function.

        Args:
            k_neighbors (int): The number of neighbors in KNN model.
            task (AttributionTask): The task to be attributed. Used to
                pass the model and hook information in this attributor.
                Please refer to the `AttributionTask` for more details.
            distance_func (Callable, optional): Customizable function
                used for distance calculation in KNN. The function
                can be quite flexible in terms of what is calculated,
                but it should take two batches of data as input.
                A typical example is as follows:
                ```python
                def f(batch_x, batch_y):
                    coord1 = batch_x[0]
                    coord2 = batch_y[0]
                    return torch.cdist(coord1, coord2)
                ```.
                If not provided, a default Euclidean distance function
                will be used.

        Raises:
            NotImplementedError: If task is not None.
        """
        self.k_neighbors = k_neighbors

        if task is not None:
            error_msg = (
                "Specifying the model via the task argument is not implemented yet."
            )
            raise NotImplementedError(error_msg)

        self.distance_func = default_dist_func
        if distance_func is not None:
            self.distance_func = distance_func

    def cache(self) -> None:
        """Precompute and cache some values for efficiency."""

    def attribute(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        train_labels: Optional[List[int]] = None,
        test_labels: Optional[List[int]] = None,
    ) -> None:
        """Calculate the KNN shapley values of the training set on each test sample.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for
                training samples to calculate the shapley values. The dataloader
                should not be shuffled.
            test_dataloader (torch.utils.data.DataLoader): The dataloader for
                test samples to calculate the shapley values. The dataloader
                should not be shuffled.
            train_labels: (List[int], optional): The list of training labels,
                with the same size and order of the training dataloader.
                If not provided, the last element in each batch from the loader
                will be used as label.
            test_labels: (List[int], optional): The list of test labels,
                with the same size and order of the test dataloader.
                If not provided, the last element in each batch from the loader
                will be used as label.

        Returns:
            Tensor: The KNN shapley values of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).

        Raises:
            ValueError: If the length of provided labels and dataset mismatch.
        """
        _check_shuffle(test_dataloader)
        _check_shuffle(train_dataloader)

        if train_labels is not None and len(train_labels) != len(
            train_dataloader.sampler,
        ):
            error_msg = "Length of training labels and training dataset mismatch."
            raise ValueError(error_msg)

        if test_labels is not None and len(test_labels) != len(test_dataloader.sampler):
            error_msg = "Length of test labels and test dataset mismatch."
            raise ValueError(error_msg)

        num_train_samples = len(train_dataloader.sampler)
        num_test_samples = len(test_dataloader.sampler)

        dist_matrix = torch.zeros(
            size=(num_test_samples, num_train_samples),
        )

        shapley_values = torch.zeros(
            size=(num_test_samples, num_train_samples),
        )

        if train_labels is None:
            train_labels = []

        if test_labels is None:
            test_labels = []

        for test_batch_idx, test_batch_data in enumerate(
            tqdm(
                test_dataloader,
                desc="fitting KNN...",
                leave=False,
            ),
        ):
            for train_batch_idx, train_batch_data in enumerate(
                train_dataloader,
            ):
                partial_dist = self.distance_func(test_batch_data, train_batch_data)

                # results position based on batch info
                col_st = train_batch_idx * train_dataloader.batch_size
                col_ed = min(
                    (train_batch_idx + 1) * train_dataloader.batch_size,
                    len(train_dataloader.sampler),
                )

                row_st = test_batch_idx * test_dataloader.batch_size
                row_ed = min(
                    (test_batch_idx + 1) * test_dataloader.batch_size,
                    len(test_dataloader.sampler),
                )

                dist_matrix[row_st:row_ed, col_st:col_ed] += partial_dist

                # Take the last element of the batch as the label
                # If no labels are provided.
                if len(train_labels) != len(train_dataloader.sampler):
                    train_labels.extend(train_batch_data[-1])

            if len(test_labels) != len(test_dataloader.sampler):
                test_labels.extend(test_batch_data[-1])

        nn_sorting = torch.argsort(dist_matrix, dim=-1)

        # Recursive calculation of shapley values
        for j in tqdm(
            range(num_test_samples),
            desc="calculating shapley values...",
            leave=False,
        ):
            shapley_values[j, nn_sorting[j, -1]] = (
                train_labels[nn_sorting[j, -1]] == test_labels[j]
            ) / num_train_samples

            for i in torch.arange(num_train_samples - 2, -1, -1):
                shapley_values[j, nn_sorting[j, i]] = shapley_values[
                    j,
                    nn_sorting[j, i + 1],
                ] + (
                    int(train_labels[nn_sorting[j, i]] == test_labels[j])
                    - int(train_labels[nn_sorting[j, i + 1]] == test_labels[j])
                ) / self.k_neighbors * min([self.k_neighbors, i + 1]) / (i + 1)

        return shapley_values
