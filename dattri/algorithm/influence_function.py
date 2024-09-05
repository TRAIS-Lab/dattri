"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

    from torch import Tensor
    from torch.utils.data import DataLoader

import warnings
from functools import partial

import torch
from torch.func import grad, vmap
from tqdm import tqdm

from dattri.algorithm.utils import _check_shuffle
from dattri.func.hessian import ihvp_arnoldi, ihvp_cg, ihvp_explicit, ihvp_lissa
from dattri.func.utils import flatten_params

from .base import BaseAttributor, BaseInnerProductAttributor


def _lissa_collate_fn(
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


SUPPORTED_IHVP_SOLVER = {
    "explicit": ihvp_explicit,
    "cg": ihvp_cg,
    "arnoldi": ihvp_arnoldi,
    "lissa": partial(ihvp_lissa, collate_fn=_lissa_collate_fn),
}


class IFAttributor(BaseAttributor):
    """Influence function attributor."""

    def __init__(
        self,
        target_func: Callable,
        params: dict,
        ihvp_solver: str = "explicit",
        ihvp_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        """Influence function attributor.

        Args:
            target_func (Callable): The target function to be attributed.
                The function can be quite flexible in terms of what is calculate,
                but it should take the parameters and the dataloader as input.
                A typical example is as follows:
                ```python
                @flatten_func(model)
                def f(params, dataloader):
                    loss = nn.CrossEntropyLoss()
                    loss_val = 0
                    for image, label in dataloader:
                        yhat = torch.func.functional_call(model, params, image)
                        loss_val += loss(yhat, label)
                    return loss_val
                ```.
                This examples calculates the loss of the model on the dataloader.
            params (dict): The parameters of the target function. The key is the
                name of a parameter and the value is the parameter tensor.
                TODO: This should be changed to support a list of parameters or
                    paths for ensembling and memory efficiency.
            ihvp_solver (str): The solver for inverse hessian vector product
                calculation, currently we only support "explicit", "cg" and "arnoldi".
            ihvp_kwargs (Optional[Dict[str, Any]]): Keyword arguments for ihvp solver.
                calculation, currently we only support "explicit", "cg", "arnoldi",
                and "lissa".
            device (str): The device to run the attributor. Default is cpu.
        """
        self.target_func = target_func
        self.params = flatten_params(params)
        if ihvp_kwargs is None:
            ihvp_kwargs = {}
        self.ihvp_solver_name = ihvp_solver
        self.ihvp_solver = partial(SUPPORTED_IHVP_SOLVER[ihvp_solver], **ihvp_kwargs)
        self.device = device
        self.full_train_dataloader = None
        self.grad_func = vmap(grad(self.target_func), in_dims=(None, 1))

    def cache(self, full_train_dataloader: DataLoader) -> None:
        """Cache the dataset for inverse hessian calculation.

        Args:
            full_train_dataloader (DataLoader): The dataloader
                with full training samples for inverse hessian calculation.
        """
        self.full_train_dataloader = full_train_dataloader

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
                test samples to calculate the influence. The dataloader should not\
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

            train_batch_grad = self.grad_func(self.params, train_batch_data)

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

                test_batch_grad = self.grad_func(self.params, test_batch_data)

                ihvp = 0
                # currently full-batch is considered
                # so only one iteration
                for full_data_ in self.full_train_dataloader:
                    # move to device
                    full_data = tuple(data.to(self.device) for data in full_data_)

                    if self.ihvp_solver_name == "lissa":
                        self.ihvp_func = self.ihvp_solver(
                            self.target_func,
                        )
                        ihvp += self.ihvp_func(
                            (self.params, *full_data),
                            test_batch_grad,
                            in_dims=(None,) + (0,) * len(full_data),
                        ).detach()
                    else:
                        self.ihvp_func = self.ihvp_solver(
                            partial(self.target_func, data_target_pair=full_data),
                        )
                        ihvp += self.ihvp_func((self.params,), test_batch_grad).detach()

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

                tda_output[row_st:row_ed, col_st:col_ed] += train_batch_grad @ ihvp.T

        return tda_output


class IFAttributorExplicit(BaseInnerProductAttributor):
    """The inner product attributor with explicit inverse hessian transformation."""

    def transformation_on_query(
        self,
        index: int,
        train_data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ihvp_explicit.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            train_data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query to be transformed. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.hessian import ihvp_explicit

        self.ihvp_func = ihvp_explicit(
            partial(self.task.get_loss_func(), data_target_pair=train_data),
            **transformation_kwargs,
        )
        model_params, _ = self.task.get_param(index)
        return self.ihvp_func((model_params,), query).detach()


class IFAttributorCG(BaseInnerProductAttributor):
    """The inner product attributor with CG inverse hessian transformation."""

    def transformation_on_query(
        self,
        index: int,
        train_data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ihvp_cg.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            train_data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query to be transformed. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.hessian import ihvp_cg

        self.ihvp_func = ihvp_cg(
            partial(self.task.get_loss_func(), data_target_pair=train_data),
            **transformation_kwargs,
        )
        model_params, _ = self.task.get_param(index)
        return self.ihvp_func((model_params,), query).detach()


class IFAttributorArnoldi(BaseInnerProductAttributor):
    """The inner product attributor with Arnoldi inverse hessian transformation."""

    def transformation_on_query(
        self,
        index: int,
        train_data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ihvp_arnoldi.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            train_data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query to be transformed. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.projection import arnoldi_project

        if not hasattr(self, "arnoldi_projector"):
            feature_dim = query.shape[1]
            func = partial(self.task.get_loss_func(), data_target_pair=train_data)
            model_params, _ = self.task.get_param(index)
            self.arnoldi_projector = arnoldi_project(
                feature_dim,
                func,
                model_params,
                device=self.device,
                **transformation_kwargs,
            )

        return self.arnoldi_projector(query).detach()


class IFAttributorLiSSA(BaseInnerProductAttributor):
    """The inner product attributor with LiSSA inverse hessian transformation."""

    def transformation_on_query(
        self,
        index: int,
        train_data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ihvp_lissa.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            train_data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query to be transformed. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.hessian import ihvp_lissa

        self.ihvp_func = ihvp_lissa(
            self.task.get_loss_func(),
            collate_fn=IFAttributorLiSSA.lissa_collate_fn,
            **transformation_kwargs,
        )
        model_params, _ = self.task.get_param(index)
        return self.ihvp_func(
            (model_params, *train_data),
            query,
            in_dims=(None,) + (0,) * len(train_data),
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
    """The attributor using DataInf."""

    def transformation_on_query(
        self,
        index: int,
        train_data: Tuple[torch.Tensor, ...],
        query: torch.Tensor,
        **transformation_kwargs,
    ) -> torch.Tensor:
        """Calculate the transformation on the query through ifvp_datainf.

        Args:
            index (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            train_data (Tuple[Tensor]): The train data. Normally this is a tuple
                of input data and target data, the number of items in the
                tuple should be aligned in the target function. The tensors'
                shape follows (1, batchsize, ...).
            query (torch.Tensor): The query to be transformed. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        model_params, param_layer_map = self.task.get_param(index, layer_split=True)
        
        def _single_datainf(
            v: torch.Tensor,
            grad: torch.Tensor,
            regularization: float,
        ) -> torch.Tensor:
            """Intermediate DataInf value calculation of a single data point.

            Args:
                v (torch.Tensor): A tensor representing (batched) validation
                    set gradient,Normally of shape (num_valiation,parameter_size)
                grad (torch.Tensor): A tensor representing a single training
                    gradient, of shape (parameter_size,)
                q (torch.Tensor): A tensor representing (batched) training gradient
                    , Normally of shape (num_train,parameter_size)
                regularization (float): A float or list of floats default
                    to 0.0.Specifies the regularization term to be added to
                    the Hessian matrix in each layer.

            Returns:
                A tensor corresponding to intermediate influence value. This value
                    will later to aggregated to obtain final influence.
            """
            grad = grad.unsqueeze(-1)
            coef = (v @ grad) / (regularization + torch.norm(grad) ** 2)
            return (v - coef @ grad.T) / regularization
        
        regularization = None
        train_batch_data = tuple(
            data.to(self.device).unsqueeze(0) for data in train_data
        )
        train_batch_grad = self.generate_train_query(
            index=self.index,
            data=train_batch_data,
        )
        query_split = self.get_layer_wise_grads(query)
        train_grad_split = self.get_layer_wise_grads(train_batch_grad)
        layer_cnt = len(train_grad_split)
        ifvps = []
        batch_size = 16
        for layer in range(layer_cnt):
            grad_layer = train_grad_split[layer]
            length = grad_layer.shape[0]
            grad_batches = grad_layer.split(batch_size,dim=0)
            running_contributions = 0

            for batch in grad_batches:
                reg = 0.0 if regularization is None else regularization[layer]
                contribution = torch.func.vmap(
                    lambda grad, layer=layer, reg=reg: _single_datainf(
                    query_split[layer],
                    grad,
                    reg,
                )
                )(batch)
                running_contributions += contribution.sum(dim=0)
            running_contributions /= length
            ifvps.append(running_contributions)
        return torch.cat(ifvps,dim=1)

    def get_layer_wise_grads(self, query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split a gradient into layer-wise gradients.

        Args:
            query (torch.Tensor): Input gradient to split, could be train/test
                gradients. Normally of shape (batch_size,parameter)

        Returns:
            Tuple[torch.Tensor, ...]: The layer-wise splitted tensor, a tuple of shape
                (batch_size,layer0_size), (batch_size,layer1_size)...
        """
        model_params, param_layer_map = self.task.get_param(
            self.index, layer_split=True,
        )
        split_index = [0] * (param_layer_map[-1] + 1)
        for idx, layer_index in enumerate(param_layer_map):
            split_index[layer_index] += model_params[idx].shape[0]
        current_idx = 0
        query_split = []
        for i in range(len(split_index)):
            query_split.append(query[:, current_idx : split_index[i] + current_idx])
            current_idx += split_index[i]
        return query_split

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

                train_grad_split = self.get_layer_wise_grads(train_batch_grad)

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
