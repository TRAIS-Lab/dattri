"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple

    from torch import Tensor
    from torch.utils.data import DataLoader

    from dattri.task import AttributionTask

import warnings
from abc import abstractmethod
from functools import partial

import torch
from tqdm import tqdm

from .base import BaseAttributor
from .utils import _check_shuffle


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
                the transformation function.
            device (str): The device to run the attributor.
        """
        self.task = task
        self.transformation_kwargs = transformation_kwargs or {}
        self.device = device
        self.index = 0

    def _set_test_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set test dataloader."""
        # This function may be overrided by the subclass
        self.test_dataloader = dataloader

    def _set_train_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set train dataloader to be attributed."""
        # This function may be overrided by the subclass
        self.train_dataloader = dataloader

    def _set_full_train_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Set train dataloader for full training set."""
        # This function may be overrided by the subclass
        self.full_train_dataloader = dataloader

    def generate_test_query(
        self,
        index: int,
        data: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Calculating the query based on the test data.

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
        return self.task.get_grad_target_func()(model_params, data)

    @abstractmethod
    def transformation_on_query(
        self,
        index: int,
        data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the transformation on the query.

        Inner product attributor calculates the inner product between the
        test query and the transformation of the test query. This function
        calculates the transformation of the test query based on the train data.

        Normally speaking, this function may return any transformation on the query.
        Hessian Matrix and Fisher Information Matrix are two common transformations.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query based on the test data. Normally it is
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
                        data=full_data,
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

        return tda_output


class IFAttributorExplicit(BaseInnerProductAttributor):
    """The inner product attributor with explicit inverse hessian transformation."""

    def transformation_on_query(
        self,
        index: int,
        data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ihvp_explicit.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query based on the test data. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.ihvp import ihvp_explicit

        self.ihvp_func = ihvp_explicit(
            partial(self.task.get_target_func(), data_target_pair=data),
            **transformation_kwargs,
        )
        model_params, _ = self.task.get_param(index)
        return self.ihvp_func((model_params,), query).detach()


class IFAttributorCG(BaseInnerProductAttributor):
    """The inner product attributor with CG inverse hessian transformation."""

    def transformation_on_query(
        self,
        index: int,
        data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ihvp_cg.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query based on the test data. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.ihvp import ihvp_cg

        self.ihvp_func = ihvp_cg(
            partial(self.task.get_target_func(), data_target_pair=data),
            **transformation_kwargs,
        )
        model_params, _ = self.task.get_param(index)
        return self.ihvp_func((model_params,), query).detach()


class IFAttributorArnoldi(BaseInnerProductAttributor):
    """The inner product attributor with Arnoldi inverse hessian transformation."""

    def transformation_on_query(
        self,
        index: int,
        data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ihvp_arnoldi.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query based on the test data. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.ihvp import ihvp_arnoldi

        self.ihvp_func = ihvp_arnoldi(
            partial(self.task.get_target_func(), data_target_pair=data),
            **transformation_kwargs,
        )
        model_params, _ = self.task.get_param(index)
        return self.ihvp_func((model_params,), query).detach()


class IFAttributorLiSSA(BaseInnerProductAttributor):
    """The inner product attributor with LiSSA inverse hessian transformation."""

    def transformation_on_query(
        self,
        index: int,
        data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ihvp_lissa.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query based on the test data. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.ihvp import ihvp_lissa

        self.ihvp_func = ihvp_lissa(
            self.task.get_target_func(),
            collate_fn=IFAttributorLiSSA.lissa_collate_fn,
            **transformation_kwargs,
        )
        model_params, _ = self.task.get_param(index)
        return self.ihvp_func(
            (model_params, *data),
            query,
            in_dims=(None,) + (0,) * len(data),
        ).detach()

    @staticmethod
    def lissa_collate_fn(
        sampled_input: List[Tensor],
    ) -> Tuple[Tensor, List[Tuple[Tensor, ...]]]:
        """Collate function for LISSA.

        Args:
            sampled_input (List[Tensor]): The sampled input from the dataloader.

        Returns:
            Tuple[Tensor, List[Tuple[Tensor, ...]]]: The collated input for the LISSA.
        """
        return (
            sampled_input[0],
            tuple(sampled_input[i].float() for i in range(1, len(sampled_input))),
        )


class IFAttributorDataInf(BaseInnerProductAttributor):
    """The inner product attributor with DataInf inverse hessian transformation."""

    def transformation_on_query(
        self,
        index: int,
        data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ihvp_datainf.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query based on the test data. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.ihvp import ihvp_datainf

        model_params, param_layer_map = self.task.get_param(index, layer_split=True)

        self.ihvp_func = ihvp_datainf(
            self.task.get_target_func(),
            0,
            (None, 0),
            param_layer_map=param_layer_map,
            **transformation_kwargs,
        )

        split_index = [0] * (param_layer_map[-1] + 1)
        for idx, layer_index in enumerate(param_layer_map):
            split_index[layer_index] += model_params[idx].shape[0]

        current_idx = 0
        query_split = []
        for i in range(len(split_index)):
            query_split.append(query[:, current_idx : split_index[i] + current_idx])
            current_idx += split_index[i]

        res = self.ihvp_func(
            (model_params, (data[0], data[1].view(-1, 1).float())),
            query_split,
        )
        return torch.cat(res, dim=1).detach()
