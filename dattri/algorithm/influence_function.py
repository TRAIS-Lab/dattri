"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import List, Optional, Tuple, Union

    from torch import Tensor
    from torch.utils.data import DataLoader

import warnings
from functools import partial

import torch
from tqdm import tqdm

from .base import BaseInnerProductAttributor
from .utils import _check_shuffle


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
    train_grads: torch.Tensor,
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
            train_grads: (torch.Tensor): The training data gradients for influence
                calculation. The shape follows (batchsize,num_parameters).
            query (torch.Tensor): The query to be transformed. Normally it is
                a 2-d dimensional tensor with the shape of
                (batchsize, num_parameters).
            transformation_kwargs (Dict[str, Any]): The keyword arguments for
                the transformation function.

        Returns:
            torch.Tensor: The transformation on the query. Normally it is a 2-d
                dimensional tensor with the shape of (batchsize, transformed_dimension).
        """
        from dattri.func.fisher import ifvp_datainf

        model_params, param_layer_map = self.task.get_param(index, layer_split=True)
        
        self.ihvp_func = ifvp_datainf(
            self.task.get_loss_func(),
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
            (model_params, (train_data[0], train_data[1].view(-1, 1).float())),
            query_split,
        )
        return torch.cat(res, dim=1).detach()

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

    @staticmethod
    def datainf(
            func: Callable,
            argnums: int,
            in_dims: Tuple[Union[None, int], ...],
            regularization: Optional[Union[float, List[float]]] = None,
            param_layer_map: [List[int]] = None,
        ) -> Callable:
        """DataInf algorithm function.

        Returns a function that,when given vectors, computes the DataInf influence.

        DataInf assume the loss to be cross-entropy and thus derive a closed form
        IFVP without having to approximate the FIM. Implementation for reference:
        https://github.com/ykwon0407/DataInf/blob/main/src/influence.py

        Args:
            func (Callable): A Python function that takes one or more arguments.
                Must return a single-element Tensor. The layer-wise gradients will
                be calculated on this function. Note that datainf expects the loss
                to be cross-entropy.
            argnums (int): An integer default to 0. Specifies which argument of func
                to compute inverse FIM with respect to.
            in_dims (Tuple[Union[None, int], ...]): Parameter sent to vmap to
                produce batched layer-wise gradients. Example: inputs,
                weights, labels corresponds to (0,None,0).
            regularization (List [float]): A float or list of floats default to 0.0.
                Specifies the regularization term to be added to the Hessian
                matrix in each layer. This is useful when the Hessian matrix is
                singular or ill-conditioned. The regularization term is
                `regularization * I` , where `I` is the identity matrix directly
                added to the Hessian matrix. The list is of length L, where L is
                the total number of layers.
            param_layer_map: Optional[List[int]]: Specifies how the parameters
                are grouped into layers. Should be the same length as parameters
                tuple. For example,for a two layer model, params =
                (0.weights1,0.bias,1.weights,1.bias),param_layer_map should be
                [0,0,1,1],resulting in two layers as expected.

        Returns:
            A function that takes a list of tuples of Tensor `x`, a tuple of tensors
                `v`(layer-wise), and a tuple of tensors `q`(layer-wise)
                and returns the approximated influence.

        Raises:
            DataInfUsageError: If the length of regularization is not the same
                as the number of layers.
        """
        class DataInfUsageError(Exception):
            """The usage exception class for DataInf."""
        batch_grad_func = torch.func.vmap(
            torch.func.grad(func, argnums=argnums), in_dims=in_dims,
        )
        if regularization is not None and not isinstance(regularization, list):
            regularization = [regularization] * len(param_layer_map)

        if param_layer_map is not None and len(regularization) != len(
            param_layer_map,
        ):
            error_msg = "The length of regularization should\
                        be the same as the number of layers."
            raise DataInfUsageError(error_msg)

        def _single_datainf(
            v: torch.Tensor,
            grad: torch.Tensor,
            q: torch.Tensor,
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
            return (q @ v.T - (q @ grad) @ coef.T) / regularization

        def _datainf_func(
            x: Tuple[torch.Tensor, ...],
            v: Tuple[torch.Tensor, ...],
            q: Tuple[torch.Tensor, ...],
        ) -> Tuple[torch.Tensor]:
            """The influence function using DataInf.

            Args:
                x (Tuple[torch.Tensor, ...]): The function will compute the
                    inverse FIM with respect to these arguments.
                v (Tuple[torch.Tensor, ...]): Tuple of layer-wise tensors from which
                    influence will be computed. For example layer-wise gradients
                    of test samples.
                q (Tuple[torch.Tensor, ...]): Tuple of layer-wise tensors from which
                    influence will be computed. For example layer-wise gradients
                    of train samples.

            Returns:
                Layer-wise influence value of shape (train_size,validation_size).
            """
            grads = batch_grad_func(*x)
            layer_cnt = len(grads)
            if param_layer_map is not None:
                grouped = []
                max_layer = max(param_layer_map)
                for group in range(max_layer + 1):
                    grouped_layers = tuple(
                        grads[layer]
                        for layer in range(len(param_layer_map))
                        if param_layer_map[layer] == group
                    )
                    concated_grads = torch.concat(grouped_layers, dim=1)
                    grouped.append(concated_grads)
                grads = tuple(grouped)
                layer_cnt = max_layer + 1  # Assuming count starts from 0
            ifvps = []
            for layer in range(layer_cnt):
                grad_layer = grads[layer]
                reg = 0.0 if regularization is None else regularization[layer]
                print(len(grad_layer))
                ifvp_contributions = torch.func.vmap(
                    lambda grad, layer=layer, reg=reg: _single_datainf(
                        v[layer],
                        grad,
                        q[layer],
                        reg,
                    ),
                )(grad_layer)
                ifvp_at_layer = ifvp_contributions.mean(dim=0)
                ifvps.append(ifvp_at_layer)
            return tuple(ifvps)

        return _datainf_func

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
            model_params, param_layer_map = self.task.get_param(
                self.index, layer_split=True,
            )
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

                    query_split = self.get_layer_wise_grads(test_batch_grad)

                    for full_data_ in self.full_train_dataloader:
                        # move to device
                        full_data = tuple(data.to(self.device) for data in full_data_)
                        inf_func = self.datainf(
                            self.task.get_loss_func(),
                            0,
                            (None, 0),
                            param_layer_map=param_layer_map,
                            **self.transformation_kwargs,
                        )
                        single_influence = inf_func(
                            (
                                model_params,
                                (full_data[0], full_data[1].view(-1, 1).float()),
                            ),
                            query_split,
                            train_grad_split,
                        )
                    print(f"Single influence len: {len(single_influence)}")
                    print(f"Shape: {single_influence[0].shape}")
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
                    influence = torch.stack(single_influence, dim=0)
                    average_influence = torch.mean(influence, dim=0)
                    tda_output[row_st:row_ed, col_st:col_ed] += average_influence

        tda_output /= checkpoint_running_count

        return tda_output
