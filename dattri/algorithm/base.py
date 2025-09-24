"""This module implement the attributor base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Union

    import torch
    from torch.utils.data import DataLoader

    from dattri.task import AttributionTask

import warnings

import torch
from tqdm import tqdm

from .utils import _check_shuffle


class BaseAttributor(ABC):
    """Base class for all attributors."""

    @abstractmethod
    def __init__(self, **kwargs: dict) -> None:
        """Initialize the attributor.

        Args:
            **kwargs (dict): Keyword arguments for the attributor.

        Returns:
            None.
        """

    @abstractmethod
    def cache(self, full_train_dataloader: torch.utils.data.DataLoader) -> None:
        """Precompute and cache values for efficiency.

        Args:
            full_train_dataloader (torch.utils.data.DataLoader): Dataloader for
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
        """Attribute the influence of training data on test data.

        Args:
            train_dataloader (torch.utils.data.DataLoader): Dataloader for
                the training data.
            test_dataloader (torch.utils.data.DataLoader): Dataloader for
                the test data.

        Returns:
            torch.Tensor: The influence of the training data on the test data.
        """


class BaseInnerProductAttributor(BaseAttributor):
    """Base class for inner product attributors."""

    def __init__(
        self,
        task: AttributionTask,
        layer_name: Optional[Union[str, List[str]]] = None,
        device: Optional[str] = "cpu",
    ) -> None:
        """Initialize the attributor.

        Args:
            task (AttributionTask): The attribution task. Must be an instance of
                `AttributionTask`.
            layer_name (Optional[Union[str, List[str]]]): The name of the layer to be
                used to calculate the train/test representations. If None, full
                parameters are used. This should be a string or a list of strings
                if multiple layers are needed. The name of layer should follow the
                key of model.named_parameters(). Default: None.
            device (str): The device to run the attributor on.
        """
        self.task = task
        self.device = device
        self.layer_name = layer_name
        self.full_train_dataloader = None

    def _set_test_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set test dataloader.

        Args:
            dataloader (DataLoader): Dataloader for test samples to be attributed.
        """
        # This function may be overridden by the subclass
        self.test_dataloader = dataloader

    def _set_train_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set train dataloader to be attributed.

        Args:
            dataloader (DataLoader): Dataloader for train samples to be attributed.
        """
        # This function may be overridden by the subclass
        self.train_dataloader = dataloader

    def _set_full_train_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set train dataloader for full training set.

        Args:
            dataloader (DataLoader): Dataloader for full training samples.
        """
        # This function may be overridden by the subclass
        self.full_train_dataloader = dataloader

    def generate_test_rep(
        self,
        ckpt_idx: int,
        data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Generate initial representations of test data.

        Inner product attributors calculate the inner product between the (transformed)
        train representations and test representations. This function generates the
        initial test representations.

        The default implementation calculates the gradient of the test loss with respect
        to the parameter. Subclasses may override this function to calculate something
        else.

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
        model_params, _ = self.task.get_param(ckpt_idx, layer_name=self.layer_name)
        return self.task.get_grad_target_func(
            layer_name=self.layer_name,
            ckpt_idx=ckpt_idx,
        )(model_params, data)

    def generate_train_rep(
        self,
        ckpt_idx: int,
        data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Generate initial representations of train data.

        Inner product attributors calculate the inner product between the (transformed)
        train representations and test representations. This function generates the
        initial train representations.

        The default implementation calculates the gradient of the train loss with
        respect to the parameter. Subclasses may override this function to
        calculate something else.

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
        model_params, _ = self.task.get_param(ckpt_idx, layer_name=self.layer_name)
        grad_loss_func = self.task.get_grad_loss_func(
            layer_name=self.layer_name,
            ckpt_idx=ckpt_idx,
        )
        return grad_loss_func(model_params, data)

    def transform_test_rep(  # noqa: PLR6301
        self,
        ckpt_idx: int,  # noqa:ARG002
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
        return test_rep

    def transform_train_rep(  # noqa: PLR6301
        self,
        ckpt_idx: int,  # noqa:ARG002
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
        return train_rep

    def cache(self, full_train_dataloader: torch.utils.data.DataLoader) -> None:
        """Cache the full training dataloader or precompute and cache more information.

        By default, the cache function only caches the full training dataloader.
        Subclasses may override this function to precompute and cache more information.

        Args:
            full_train_dataloader (torch.utils.data.DataLoader): Dataloader for
                the full training data. Ideally, the batch size of the dataloader
                should be the same as the number of training samples to get the
                best accuracy for some attributors. Smaller batch size may lead to
                a less accurate result but lower memory consumption.
        """
        self._set_full_train_data(full_train_dataloader)

    def attribute(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        relatif_method: Optional[str] = None,
    ) -> torch.Tensor:
        """Calculate the influence of the training set on the test set.

        Args:
            train_dataloader (DataLoader): Dataloader for training samples to
                calculate the influence. It can be a subset of the full training
                set if `cache` is called before. A subset means that only a part
                of the training set's influence is calculated. The dataloader should
                not be shuffled.
            test_dataloader (DataLoader): Dataloader for test samples to calculate
                the influence. The dataloader should not be shuffled.
            relatif_method (Optional[str]): Method for normalizing the
                influence values.
                Supported options:
                - `"l"`: Normalizes by `sqrt(g_i^T (H^-1 g_i))`.
                - `"theta"`: Normalizes by `||H^-1 g_i||`.
                - `None`: No normalization applied.

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

                denom = None
                if relatif_method is not None:
                    if relatif_method == "l":
                        test_batch_rep = self.generate_test_rep(
                            ckpt_idx=checkpoint_idx,
                            data=train_batch_data,
                        )
                    else:
                        test_batch_rep = None
                    denom = self._compute_denom(
                        checkpoint_idx,
                        train_batch_rep,
                        test_batch_rep,
                        relatif_method=relatif_method,
                    )

                # transform the train representations
                train_batch_rep = self.transform_train_rep(
                    ckpt_idx=checkpoint_idx,
                    train_rep=train_batch_rep,
                )

                for test_batch_idx, test_batch_data_ in enumerate(
                    tqdm(
                        test_dataloader,
                        desc="calculating gradient of test set...",
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
                        train_batch_rep @ test_batch_rep.T / denom.unsqueeze(-1)
                        if denom is not None
                        else train_batch_rep @ test_batch_rep.T
                    )

            tda_output /= checkpoint_idx + 1

        return tda_output

    def _compute_denom(
        self,
        ckpt_idx: int,  # noqa: ARG002
        train_batch_rep: torch.Tensor,
        test_batch_rep: Optional[torch.Tensor] = None,
        relatif_method: Optional[str] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute the denominator for the influence calculation.

        Args:
            ckpt_idx (int): The index of the checkpoint being used for influence
                calculation.
            train_batch_rep (torch.Tensor): The representation of the training batch
                at the given checkpoint.
            test_batch_rep (Optional[torch.Tensor]): The representation of the
                training batch, generated using `generate_test_rep` at the given
                checkpoint.
            relatif_method (Optional[str]): Normalization method.
                - `"l"`: Computes `sqrt(g_i^T (H^-1 g_i))`.
                - `"theta"`: Computes `||H^-1 g_i||`.
                - `None`: Raises an error.

        Returns:
            torch.Tensor: The computed denominator for normalization. It is a
            1-d dimensional tensor with the shape of (batch_size).
        """
        _ = self
        _ = test_batch_rep

        batch_size = train_batch_rep.size(0)
        return train_batch_rep.new_ones(batch_size)
