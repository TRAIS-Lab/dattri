"""This module implement the attributor base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, Optional, Tuple

    import torch
    from torch.utils.data import DataLoader

    from dattri.task import AttributionTask

import warnings

import torch
from tqdm import tqdm

from .utils import _check_shuffle


class BaseAttributor(ABC):
    """BaseAttributor."""

    @abstractmethod
    def __init__(self, **kwargs: dict) -> None:
        """Initialize the attributor.

        Args:
            **kwargs (dict): The keyword arguments for the attributor.

        Returns:
            None.
        """

    @abstractmethod
    def cache(self, full_train_dataloader: torch.utils.data.DataLoader) -> None:
        """Precompute and cache some values for efficiency.

        Args:
            full_train_dataloader (torch.utils.data.DataLoader): The dataloader for
                the training data.

        Returns:
            None.
        """

    @abstractmethod
    def attribute(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Attribute the influence of the training data on the test data.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for
                the training data.
            test_dataloader (torch.utils.data.DataLoader): The dataloader for
                the test data.

        Returns:
            torch.Tensor: The influence of the training data on the test data.
        """


class BaseInnerProductAttributor(BaseAttributor):
    """The base class for inner product attributor."""

    def __init__(
        self,
        task: AttributionTask,
        device: Optional[str] = "cpu",
        **transformation_kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the attributor.

        Args:
            task (AttributionTask): The task to be attributed. The task should
                be an instance of `AttributionTask`.
            transformation_kwargs (Optional[Dict[str, Any]]): The keyword arguments for
                the transformation function. More specifically, it will be stored in
                the `transformation_kwargs` attribute and be used by some customized
                member functions, e.g., `transformation_on_query`, where the
                transformation such as hessian matrix or Fisher Information matrix is
                calculated.
            device (str): The device to run the attributor.
        """
        self.task = task
        self.transformation_kwargs = transformation_kwargs or {}
        self.device = device
        self.index = 0

    def _set_test_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set test dataloader.

        Args:
            dataloader (DataLoader): The dataloader for test samples to be attributed.
        """
        # This function may be overrided by the subclass
        self.test_dataloader = dataloader

    def _set_train_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set train dataloader to be attributed.

        Args:
            dataloader (DataLoader): The dataloader for train samples to be attributed.
        """
        # This function may be overrided by the subclass
        self.train_dataloader = dataloader

    def _set_full_train_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set train dataloader for full training set.

        Args:
            dataloader (DataLoader): The dataloader for full training samples.
        """
        # This function may be overrided by the subclass
        self.full_train_dataloader = dataloader

    def generate_test_query(
        self,
        index: int,
        data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Calculating the query based on the test data.

        Inner product attributor calculates the inner product between the
        train query and the transformation of the test query. This function
        calculates the test query based on the test data.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model
                parameters.
            data (Tuple[Tensor]): The test data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).

        Returns:
            torch.Tensor: The query based on the test data. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
        """
        model_params, _ = self.task.get_param(index)
        return self.task.get_grad_target_func()(model_params, data)

    def generate_train_query(
        self,
        index: int,
        data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Calculating the query based on the train data.

        Inner product attributor calculates the inner product between the
        train query and the transformation of the test query. This function
        calculates the train query based on the train data.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model
                parameters.
            data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).

        Returns:
            torch.Tensor: The query based on the train data. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
        """
        model_params, _ = self.task.get_param(index)
        return self.task.get_grad_loss_func()(model_params, data)

    @abstractmethod
    def transformation_on_query(
        self,
        index: int,
        train_data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the transformation on the query.

        Inner product attributor calculates the inner product between the
        train query and the transformation of the test query. This function
        calculates the transformation of the test query.

        Normally speaking, this function may return any transformation on the query.
        Hessian Matrix and Fisher Information Matrix are two common transformations.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            train_data (Tuple[Tensor]): Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query to be transformed. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).

        Returns:
            torch.Tensor: The transformation on the query. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, transformed_dimension).
        """

    def cache(self, full_train_dataloader: DataLoader) -> None:
        """Cache the dataset for inverse hessian calculation.

        Args:
            full_train_dataloader (DataLoader): The dataloader
                with full training samples for inverse hessian calculation.
        """
        self._set_full_train_data(full_train_dataloader)

    def attribute(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> torch.Tensor:
        """Calculate the influence of the training set on the test set.

        Args:
            train_dataloader (DataLoader): The dataloader for
                training samples to calculate the influence. It can be a subset
                of the full training set if `cache` is called before. A subset
                means that only a part of the training set's influence is calculated.
                The dataloader should not be shuffled.
            test_dataloader (DataLoader): The dataloader for
                test samples to calculate the influence. The dataloader should not
                be shuffled.

        Returns:
            torch.Tensor: The influence of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).

        """
        super().attribute(train_dataloader, test_dataloader)
        if self.full_train_dataloader is None:
            self.full_train_dataloader = train_dataloader
            warnings.warn(
                "The full training data loader was NOT cached. \
                           Treating the train_dataloader as the full training \
                           data loader.",
                stacklevel=1,
            )

        _check_shuffle(train_dataloader)
        _check_shuffle(test_dataloader)

        tda_output = torch.zeros(
            size=(len(train_dataloader.sampler), len(test_dataloader.sampler)),
            device=self.device,
        )

        # TODO: sometimes the train dataloader could be swapped with the test dataloader
        # prepare a checkpoint-specific seed
        checkpoint_running_count = 0
        for checkpoint_idx in range(len(self.task.get_checkpoints())):
            # set index to current one
            self.index = checkpoint_idx
            checkpoint_running_count += 1
            tda_output *= checkpoint_running_count - 1
            for train_batch_idx, train_batch_data_ in enumerate(
                tqdm(
                    train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
                ),
            ):
                # get gradient of train
                train_batch_data = tuple(
                    data.to(self.device).unsqueeze(0) for data in train_batch_data_
                )

                train_batch_grad = self.generate_train_query(
                    index=self.index,
                    data=train_batch_data,
                )

                for test_batch_idx, test_batch_data_ in enumerate(
                    tqdm(
                        test_dataloader,
                        desc="calculating gradient of training set...",
                        leave=False,
                    ),
                ):
                    # get gradient of test
                    test_batch_data = tuple(
                        data.to(self.device).unsqueeze(0) for data in test_batch_data_
                    )

                    test_batch_grad = self.generate_test_query(
                        index=self.index,
                        data=test_batch_data,
                    )

                    vector_product = 0
                    for full_data_ in self.full_train_dataloader:
                        # move to device
                        full_data = tuple(data.to(self.device) for data in full_data_)
                        vector_product += self.transformation_on_query(
                            index=self.index,
                            train_data=full_data,
                            query=test_batch_grad,
                            **self.transformation_kwargs,
                        )

                    # results position based on batch info
                    row_st = train_batch_idx * train_dataloader.batch_size
                    row_ed = min(
                        (train_batch_idx + 1) * train_dataloader.batch_size,
                        len(train_dataloader.sampler),
                    )

                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.sampler),
                    )

                    tda_output[row_st:row_ed, col_st:col_ed] += (
                        train_batch_grad @ vector_product.T
                    )

        tda_output /= checkpoint_running_count

        return tda_output
