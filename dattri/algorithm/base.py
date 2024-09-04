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
                the full training data.

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
                member functions, e.g., `transform_test_rep`, where the
                transformation such as hessian matrix or Fisher Information matrix is
                calculated.
            device (str): The device to run the attributor.
        """
        self.task = task
        self.transformation_kwargs = transformation_kwargs or {}
        self.device = device

    def _set_test_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set test dataloader.

        Args:
            dataloader (DataLoader): The dataloader for test samples to be attributed.
        """
        # This function may be overridden by the subclass
        self.test_dataloader = dataloader

    def _set_train_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set train dataloader to be attributed.

        Args:
            dataloader (DataLoader): The dataloader for train samples to be attributed.
        """
        # This function may be overridden by the subclass
        self.train_dataloader = dataloader

    def _set_full_train_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set train dataloader for full training set.

        Args:
            dataloader (DataLoader): The dataloader for full training samples.
        """
        # This function may be overridden by the subclass
        self.full_train_dataloader = dataloader

    def generate_test_rep(
        self,
        ckpt_idx: int,
        data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Getting the initial representations of the test data.

        Inner product attributor calculates the inner product between the (transformed)
        train representations and test representations. This function generates the
        initial test representations.

        Args:
            ckpt_idx (int): The index of the model checkpoints. This index
                is used for ensembling different trained model checkpoints.
            data (Tuple[Tensor]): The test data. Typically, this is a tuple
                of input data and target data but the number of items in this
                tuple should align with the corresponding argument in the
                target function. The tensors' shape follows (1, batch_size, ...).

        Returns:
            torch.Tensor: The initial representations of the test data. Typically,
                it is a 2-d dimensional tensor with the shape of
                (batch_size, num_parameters).
        """
        model_params, _ = self.task.get_param(ckpt_idx)
        return self.task.get_grad_target_func()(model_params, data)

    def generate_train_rep(
        self,
        ckpt_idx: int,
        data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Generate the initial representations of the train data.

        Inner product attributor calculates the inner product between the (transformed)
        train representations and test representations. This function generates the
        initial train representations.

        Args:
            ckpt_idx (int): The index of the model checkpoints. This index
                is used for ensembling different trained model checkpoints.
            data (Tuple[Tensor]): The train data. Typically, this is a tuple
                of input data and target data but the number of items in this
                tuple should align with the corresponding argument in the
                target function. The tensors' shape follows (1, batch_size, ...).

        Returns:
            torch.Tensor: The initial representations of the train data. Typically,
                it is a 2-d dimensional tensor with the shape of
                (batch_size, num_parameters).
        """
        model_params, _ = self.task.get_param(ckpt_idx)
        return self.task.get_grad_loss_func()(model_params, data)

    @abstractmethod
    def transform_test_rep(
        self,
        ckpt_idx: int,
        test_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Transform the test representations.

        Inner product attributor calculates the inner product between the (transformed)
        train representations and test representations. This function calculates the
        transformation of the test representations. For example, the transformation
        could be the product of the test representations and the inverse Hessian matrix.

        Args:
            ckpt_idx (int): The index of the model checkpoints. This index
                is used for ensembling different trained model checkpoints.
            test_rep (torch.Tensor): The test representations to be transformed.
                Typically, it is a 2-d dimensional tensor with the shape of
                (batch_size, num_parameters).

        Returns:
            torch.Tensor: The transformed test representations. Typically,
                it is a 2-d dimensional tensor with the shape of
                (batch_size, transformed_dimension).
        """

    @abstractmethod
    def transform_train_rep(
        self,
        ckpt_idx: int,
        train_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Transform the train representations.

        Inner product attributor calculates the inner product between the (transformed)
        train representations and test representations. This function calculates the
        transformation of the train representations. For example, the transformation
        could be a dimension reduction of the train representations.

        Args:
            ckpt_idx (int): The index of the model checkpoints. This index
                is used for ensembling different trained model checkpoints.
            train_rep (torch.Tensor): The train representations to be transformed.
                Typically, it is a 2-d dimensional tensor with the shape of
                (batch_size, num_parameters).

        Returns:
            torch.Tensor: The transformed train representations. Typically,
                it is a 2-d dimensional tensor with the shape of
                (batch_size, transformed_dimension).
        """

    def cache(self, full_train_dataloader: torch.utils.data.DataLoader) -> None:
        """Cache the full training dataloader.

        By default, the cache function only caches the full training dataloader.
        Subclasses may override this function to precompute and cache more information.

        Args:
            full_train_dataloader (torch.utils.data.DataLoader): The dataloader for
                the full training data.

        Returns:
            None.
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
        for checkpoint_idx in range(len(self.task.get_checkpoints())):
            tda_output *= checkpoint_idx
            for train_batch_idx, train_batch_data_ in enumerate(
                tqdm(
                    train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
                ),
            ):
                # move to device
                train_batch_data = tuple(
                    data.to(self.device).unsqueeze(0) for data in train_batch_data_
                )
                # get initial representations of train data
                train_batch_rep = self.generate_train_rep(
                    ckpt_idx=checkpoint_idx,
                    data=train_batch_data,
                )
                # transform the train representations
                train_batch_rep = self.transform_train_rep(
                    ckpt_idx=checkpoint_idx,
                    train_rep=train_batch_rep,
                )

                for test_batch_idx, test_batch_data_ in enumerate(
                    tqdm(
                        test_dataloader,
                        desc="calculating gradient of training set...",
                        leave=False,
                    ),
                ):
                    # move to device
                    test_batch_data = tuple(
                        data.to(self.device).unsqueeze(0) for data in test_batch_data_
                    )
                    # get initial representations of test data
                    test_batch_rep = self.generate_test_rep(
                        ckpt_idx=checkpoint_idx,
                        data=test_batch_data,
                    )
                    # transform the test representations
                    test_batch_rep = self.transform_test_rep(
                        ckpt_idx=checkpoint_idx,
                        test_rep=test_batch_rep,
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
                        train_batch_rep @ test_batch_rep.T
                    )

            tda_output /= checkpoint_idx + 1

        return tda_output
