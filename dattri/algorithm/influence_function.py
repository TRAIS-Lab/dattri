"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Tuple

    from torch import Tensor

from functools import partial

import torch

from .base import BaseInnerProductAttributor


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
    """The inner product attributor with DataInf inverse hessian transformation."""

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
