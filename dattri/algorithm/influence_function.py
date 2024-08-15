"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Tuple

    from torch import Tensor

from functools import partial

import torch

from .base import BaseInnerProductAttributor, BaseAttributor

from typing import TYPE_CHECKING
import warnings

from torch.func import grad, vmap

import time

from .utils import _check_shuffle

from tqdm import tqdm

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

class IFAttributorDataInf(BaseAttributor):
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

    def get_layer_wise_grads(self, query):
        model_params, param_layer_map = self.task.get_param(self.index, layer_split=True)
        split_index = [0] * (param_layer_map[-1] + 1)
        for idx, layer_index in enumerate(param_layer_map):
            split_index[layer_index] += model_params[idx].shape[0]
        current_idx = 0
        query_split = []
        for i in range(len(split_index)):
            query_split.append(query[:, current_idx : split_index[i] + current_idx])
            current_idx += split_index[i]
        return query_split

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


        def ifvp_datainf_revise(
            func: Callable,
            argnums: int,
            in_dims: Tuple[Union[None, int], ...],
            regularization: Optional[Union[float, List[float]]] = None,
            param_layer_map: Optional[List[int]] = None,
        ) -> Callable:
            """DataInf IFVP algorithm function.

            Standing for the inverse-FIM-vector product, returns a function that,
            when given vectors, computes the product of inverse-FIM and vector.

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
                in_dims (Tuple[Union[None, int], ...]): Parameter sent to vmap to produce
                    batched layer-wise gradients. Example: inputs, weights, labels corresponds
                    to (0,None,0).
                regularization (List [float]): A float or list of floats default to 0.0.
                    Specifies the
                    regularization term to be added to the Hessian matrix in each layer.
                    This is useful when the Hessian matrix is singular or ill-conditioned.
                    The regularization term is `regularization * I`, where `I` is the
                    identity matrix directly added to the Hessian matrix. The list is
                    of length L, where L is the total number of layers.
                param_layer_map: Optional[List[int]]: Specifies how the parameters are grouped
                    into layers. Should be the same length as parameters tuple. For example,
                    for a two layer model, params = (0.weights1,0.bias,1.weights,1.bias),
                    param_layer_map should be [0,0,1,1],resulting in two layers as expected.

            Returns:
                A function that takes a list of tuples of Tensor `x` and a tuple of tensors
                `v`(layer-wise) and returns the approximated IFVP of the approximated Hessian of
                `func` and `v`.

            Raises:
                IFVPUsageError: If the length of regularization is not the same as the number
                    of layers.
            """
            # TODO: param_layer_map should not be optional.
            #print(f"Start of DataInf calculation!")
            initial_memory = torch.cuda.memory_allocated("cuda") / 1e6
            batch_grad_func = torch.func.vmap(grad(func, argnums=argnums), in_dims=in_dims)
            final_memory = torch.cuda.memory_allocated("cuda") / 1e6 
            #print(f"Memory usage of test grad calculating: {final_memory-initial_memory}")
            if regularization is not None and not isinstance(regularization, list):
                regularization = [regularization] * len(param_layer_map)

            if param_layer_map is not None and len(regularization) != len(param_layer_map):
                error_msg = "The length of regularization should\
                            be the same as the number of layers."
                raise IFVPUsageError(error_msg)

            def _single_datainf_ifvp(
                v: torch.Tensor,
                grad: torch.Tensor,
                q: torch.Tensor,
                regularization: float,
            ) -> torch.Tensor:
                # TODO: docstring
                #print(f"v shape: {v.shape}")
                #print(f"grad shape: {grad.shape}")
                #print(f"q shape: {q.shape}")
                grad = grad.unsqueeze(-1)
                initial_memory = torch.cuda.memory_allocated("cuda") / 1e6
                coef = (v @ grad) / (regularization + torch.norm(grad)**2)
                final_memory = torch.cuda.memory_allocated("cuda") / 1e6 
                #print(f"Memory usage of coef: {final_memory-initial_memory}")
                #print(f"Coef shape: {coef.shape}")
                initial_memory = torch.cuda.memory_allocated("cuda") / 1e6
                tmp = (q @ grad) @ coef.T
                #print(f"tmp shape: {tmp.shape}")
                res = (q @ v.T - tmp) / regularization
                final_memory = torch.cuda.memory_allocated("cuda") / 1e6 
                #print(f"Memory usage of res: {final_memory-initial_memory}")
                return res

            def _ifvp_datainf_func(
                x: Tuple[torch.Tensor, ...],
                v: Tuple[torch.Tensor, ...],
                q: Tuple[torch.Tensor, ...],
            ) -> Tuple[torch.Tensor]:
                """The IFVP function using DataInf.

                Args:
                    x (Tuple[torch.Tensor, ...]): The function will compute the
                        inverse FIM with respect to these arguments.
                    v (Tuple[torch.Tensor, ...]): Tuple of layer-wise tensors from
                        which IFVP will becomputed. For example layer-wise gradients
                        of test samples.

                Returns:
                    Layer-wise IFVP values.
                """
                #print(f"New implementation!")
                initial_memory = torch.cuda.memory_allocated("cuda") / 1e6
                grads = batch_grad_func(*x)
                final_memory = torch.cuda.memory_allocated("cuda") / 1e6 
                #print(f"Memory usage of test grad calculating: {final_memory-initial_memory}")
                layer_cnt = len(grads)
                if param_layer_map is not None:
                    grouped = []
                    max_layer = max(param_layer_map)
                    for group in range(max_layer + 1):
                        grouped_layers = tuple(
                            [
                                grads[layer]
                                for layer in range(len(param_layer_map))
                                if param_layer_map[layer] == group
                            ],
                        )
                        concated_grads = torch.concat(grouped_layers, dim=1)
                        grouped.append(concated_grads)
                    grads = tuple(grouped)
                    layer_cnt = max_layer + 1  # Assuming count starts from 0
                ifvps = []
                memory_now = torch.cuda.memory_allocated("cuda") / 1e6
                #print(f"Memory before calculation: {memory_now}")
                for layer in range(layer_cnt):
                    start_time = time.time()
                    initial_memory = torch.cuda.memory_allocated("cuda") / 1e6
                    grad_layer = grads[layer]
                    reg = 0.0 if regularization is None else regularization[layer]
                    tmp = v[layer]
                    # contri = torch.zeros((tmp.shape)).to('cuda')
                    # for grad in grad_layer:
                    #     contri = contri + _single_datainf_ifvp(tmp,grad,reg)
                    # ifvp_contributions = contri
                    ifvp_contributions = torch.func.vmap(
                        lambda grad, layer=layer, reg=reg: _single_datainf_ifvp(
                            tmp,
                            grad,
                            q[layer],
                            reg,
                        ),
                    )(grad_layer)
                    final_memory = torch.cuda.memory_allocated("cuda") / 1e6 
                    #print(f"Memory usage of mapping layer{layer}: {final_memory-initial_memory}")
                    ifvp_at_layer = ifvp_contributions.mean(dim=0)
                    ifvps.append(ifvp_at_layer)
                    end_time = time.time()
                    #print(f"Time used for layer {layer}: {end_time-start_time}")
                return tuple(ifvps)

            return _ifvp_datainf_func


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
                #print(f"Shape of train grad{train_batch_grad.shape}")
                #print(type(train_batch_grad))
                train_grad_tmp = self.get_layer_wise_grads(train_batch_grad)
                #print(len(train_grad_tmp))
                #print(train_grad_tmp[0].shape)
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
                    model_params, param_layer_map = self.task.get_param(self.index, layer_split=True)
                    for full_data_ in self.full_train_dataloader:
                        # move to device
                        query_split = self.get_layer_wise_grads(test_batch_grad)
                        full_data = tuple(data.to(self.device) for data in full_data_)
                        #print("reached!")
                        inf_func = ifvp_datainf_revise(
                            self.task.get_loss_func(),
                            0,
                            (None, 0),
                            param_layer_map=param_layer_map,
                            **self.transformation_kwargs,
                        )
                        res = inf_func(
                        (model_params, (full_data[0], full_data[1].view(-1, 1).float())),
                        query_split,
                        train_grad_tmp
                        )
                        #print(f"res shape: {res[0].shape}")
                        # vector_product += self.transformation_on_query(
                        #     index=self.index,
                        #     train_data=full_data,
                        #     query=test_batch_grad,
                        #     **self.transformation_kwargs,
                        # )
                        

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
                    stacked_tensors = torch.stack(res, dim=0)  # Shape will be [num_tensors, 64, 64]
                    #print(f"stacked_tensor shape: {stacked_tensors.shape}")
# Compute the mean along the 0th dimension
                    average_tensor = torch.mean(stacked_tensors, dim=0)
                    tda_output[row_st:row_ed, col_st:col_ed] += average_tensor

        tda_output /= checkpoint_running_count

        return tda_output