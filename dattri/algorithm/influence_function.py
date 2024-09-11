"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple

    from torch import Tensor
    from torch.utils.data import DataLoader

    from dattri.task import AttributionTask

import math
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
        Tuple[Tensor, List[Tuple[Tensor, ...]]]: The collated input for the LiSSA.
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
                The function can be quite flexible in terms of what is calculated,
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

    def __init__(
        self,
        task: AttributionTask,
        device: Optional[str] = "cpu",
        regularization: float = 0.0,
    ) -> None:
        """Initialize the explicit inverse Hessian attributor.

        Args:
            task (AttributionTask): Task to attribute. Must be an instance of
                `AttributionTask`.
            device (str): Device to run the attributor on. Default is "cpu".
            regularization (float): Regularization term added to Hessian matrix.
                Useful for singular or ill-conditioned Hessian matrices.
                Added as `regularization * I`, where `I` is the identity matrix.
                Default is 0.0.
        """
        super().__init__(task, device)
        self.transformation_kwargs = {
            "regularization": regularization,
        }

    def transform_test_rep(
        self,
        ckpt_idx: int,
        test_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the transformation on the test rep through ihvp_explicit.

        Args:
            ckpt_idx (int): Index of model parameters. Used for ensembling.
            test_rep (torch.Tensor): Test representations to be transformed.
                Typically a 2-d tensor with shape (batch_size, num_parameters).

        Returns:
            torch.Tensor: Transformed test representations. Typically a 2-d
                tensor with shape (batch_size, transformed_dimension).
        """
        from dattri.func.hessian import ihvp_explicit

        vector_product = 0
        model_params, _ = self.task.get_param(ckpt_idx)
        for full_data_ in self.full_train_dataloader:
            # move to device
            full_data = tuple(data.to(self.device) for data in full_data_)
            self.ihvp_func = ihvp_explicit(
                partial(self.task.get_loss_func(), data_target_pair=full_data),
                **self.transformation_kwargs,
            )
            vector_product += self.ihvp_func((model_params,), test_rep).detach()
        return vector_product


class IFAttributorCG(BaseInnerProductAttributor):
    """The inner product attributor with CG inverse hessian transformation."""

    def __init__(
        self,
        task: AttributionTask,
        device: Optional[str] = "cpu",
        max_iter: int = 10,
        tol: float = 1e-7,
        mode: str = "rev-rev",
        regularization: float = 0.0,
    ) -> None:
        """Initialize the CG inverse Hessian attributor.

        Args:
            task (AttributionTask): The task to be attributed. Must be an instance of
                `AttributionTask`.
            device (str): Device to run the attributor on. Default is "cpu".
            max_iter (int): Maximum iterations for Conjugate Gradient Descent. Default
                is 10.
            tol (float): Convergence tolerance. Algorithm stops if residual norm < tol.
                Default is 1e-7.
            mode (str): Auto-diff mode. Options:
                - "rev-rev": Two reverse-mode auto-diffs. Better compatibility, more
                memory cost.
                - "rev-fwd": Reverse-mode + forward-mode. Memory-efficient, less
                compatible.
            regularization (float): Regularization term for Hessian vector product.
                Adding `regularization * I` to the Hessian matrix, where `I` is the
                identity matrix. Useful for singular or ill-conditioned matrices.
                Default is 0.0.
        """
        super().__init__(task, device)
        self.transformation_kwargs = {
            "max_iter": max_iter,
            "tol": tol,
            "mode": mode,
            "regularization": regularization,
        }

    def transform_test_rep(
        self,
        ckpt_idx: int,
        test_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the transformation on the test rep through ihvp_cg.

        Args:
            ckpt_idx (int): Index of the model checkpoints. Used for ensembling
                different trained model checkpoints.
            test_rep (torch.Tensor): Test representations to be transformed.
                Typically a 2-d tensor with shape (batch_size, num_parameters).

        Returns:
            torch.Tensor: Transformed test representations. Typically a 2-d
                tensor with shape (batch_size, transformed_dimension).
        """
        from dattri.func.hessian import ihvp_cg

        vector_product = 0
        model_params, _ = self.task.get_param(ckpt_idx)
        for full_data_ in self.full_train_dataloader:
            # move to device
            full_data = tuple(data.to(self.device) for data in full_data_)
            self.ihvp_func = ihvp_cg(
                partial(self.task.get_loss_func(), data_target_pair=full_data),
                **self.transformation_kwargs,
            )
            vector_product += self.ihvp_func((model_params,), test_rep).detach()
        return vector_product


class IFAttributorArnoldi(BaseInnerProductAttributor):
    """The inner product attributor with Arnoldi projection transformation."""

    def __init__(
        self,
        task: AttributionTask,
        device: Optional[str] = "cpu",
        precompute_data_ratio: float = 1.0,
        proj_dim: int = 100,
        max_iter: int = 100,
        norm_constant: float = 1.0,
        tol: float = 1e-7,
        regularization: float = 0.0,
        seed: int = 0,
    ) -> None:
        """Initialize the Arnoldi projection attributor.

        Args:
            task (AttributionTask): The task to be attributed. Must be an instance of
                `AttributionTask`.
            device (str): Device to run the attributor on. Default is "cpu".
            precompute_data_ratio (float): Ratio of full training data used to
                precompute the Arnoldi projector. Default is 1.0.
            proj_dim (int): Dimension after projection. Corresponds to number of top
                eigenvalues to keep for Hessian approximation.
            max_iter (int): Maximum iterations for Arnoldi Iteration. Default is 100.
            norm_constant (float): Constant for the norm of the projected vector.
                May need to be > 1 for large number of parameters to avoid dividing the
                projected vector by a very large normalization constant. Default is 1.0.
            tol (float): Convergence tolerance. Algorithm stops if the norm of the
                current basis vector < tol. Default is 1e-7.
            regularization (float): Regularization term for Hessian vector product.
                Adding `regularization * I` to the Hessian matrix, where `I` is the
                identity matrix. Useful for singular or ill-conditioned matrices.
                Default is 0.0.
            seed (int): Random seed for projector. Default is 0.
        """
        super().__init__(task, device)
        self.precompute_data_ratio = precompute_data_ratio
        self.proj_dim = proj_dim
        self.max_iter = max_iter
        self.norm_constant = norm_constant
        self.tol = tol
        self.regularization = regularization
        self.seed = seed

    def cache(
        self,
        full_train_dataloader: DataLoader,
    ) -> None:
        """Cache the dataset and pre-calculate the Arnoldi projector.

        Args:
            full_train_dataloader (DataLoader): Dataloader with full training data.
        """
        self.full_train_dataloader = full_train_dataloader
        self.arnoldi_projectors = []

        # Assuming that full_train_dataloader has only one batch
        iter_number = math.ceil(len(full_train_dataloader) * self.precompute_data_ratio)
        data_target_pair_list = []
        for _ in range(iter_number):
            data_target_pair_list.append(next(iter(full_train_dataloader)))  # noqa: PERF401

        # concatenate all data
        data_target_pair = data_target_pair_list[0]
        for batch_idx in range(1, len(data_target_pair_list)):
            for data_item_idx in range(len(data_target_pair_list[0])):
                data_target_pair[data_item_idx] = torch.cat(
                    (
                        data_target_pair[data_item_idx],
                        data_target_pair_list[batch_idx][data_item_idx],
                    ),
                    dim=0,
                )

        # to device (only once for all data)
        for data_item_idx in range(len(data_target_pair)):
            data_target_pair[data_item_idx] = data_target_pair[data_item_idx].to(
                self.device,
            )

        func = partial(self.task.get_loss_func(), data_target_pair=data_target_pair)

        from dattri.func.projection import arnoldi_project

        for i in range(len(self.task.get_checkpoints())):
            model_params, _ = self.task.get_param(i)
            self.arnoldi_projectors.append(
                arnoldi_project(
                    feature_dim=len(model_params),
                    func=func,
                    x=model_params,
                    proj_dim=self.proj_dim,
                    max_iter=self.max_iter,
                    norm_constant=self.norm_constant,
                    tol=self.tol,
                    regularization=self.regularization,
                    seed=self.seed,
                    device=self.device,
                ),
            )

    def transform_test_rep(
        self,
        ckpt_idx: int,
        test_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Transform the test representations via Arnoldi projection.

        Args:
            ckpt_idx (int): Index of the model checkpoints. Used for ensembling
                different trained model checkpoints.
            test_rep (torch.Tensor): Test representations to be transformed.
                A 2-d tensor with shape (batch_size, num_params).

        Returns:
            torch.Tensor: Transformed test representations. A 2-d tensor with
                shape (batch_size, proj_dim).

        Raises:
            ValueError: If the Arnoldi projector has not been cached.
        """
        if not hasattr(self, "arnoldi_projectors"):
            error_msg = "The Arnoldi projector has not been cached.\
                         Please call cache() first."
            raise ValueError(error_msg)

        return self.arnoldi_projectors[ckpt_idx](test_rep).detach()


class IFAttributorLiSSA(BaseInnerProductAttributor):
    """The inner product attributor with LiSSA inverse hessian transformation."""

    def __init__(
        self,
        task: AttributionTask,
        device: Optional[str] = "cpu",
        batch_size: int = 1,
        num_repeat: int = 1,
        recursion_depth: int = 5000,
        damping: float = 0.0,
        scaling: float = 50.0,
        mode: str = "rev-rev",
    ) -> None:
        """Initialize the LiSSA inverse Hessian attributor.

        Args:
            task (AttributionTask): The task to be attributed. Must be an instance of
                `AttributionTask`.
            device (str): Device to run the attributor on. Default is "cpu".
            batch_size (int): Batch size for LiSSA inner loop update. Default is 1.
            num_repeat (int): Number of samples of the HVP approximation to average.
                Default is 1.
            recursion_depth (int): Number of recursions to estimate each IHVP sample.
                Default is 5000.
            damping (float): Damping factor for non-convexity in LiSSA IHVP calculation.
            scaling (float): Scaling factor for convergence in LiSSA IHVP calculation.
            mode (str): Auto-diff mode. Options:
                - "rev-rev": Two reverse-mode auto-diffs. Better compatibility, more
                memory cost.
                - "rev-fwd": Reverse-mode + forward-mode. Memory-efficient, less
                compatible.
        """
        super().__init__(task, device)
        self.transformation_kwargs = {
            "batch_size": batch_size,
            "num_repeat": num_repeat,
            "recursion_depth": recursion_depth,
            "damping": damping,
            "scaling": scaling,
            "mode": mode,
        }

    def transform_test_rep(
        self,
        ckpt_idx: int,
        test_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the transformation on the test rep through ihvp_lissa.

        Args:
            ckpt_idx (int): Index of the model checkpoints. Used for ensembling
                different trained model checkpoints.
            test_rep (torch.Tensor): Test representations to be transformed.
                Typically a 2-d tensor with shape (batch_size, num_parameters).

        Returns:
            torch.Tensor: Transformed test representations. Typically a 2-d
                tensor with shape (batch_size, transformed_dimension).
        """
        from dattri.func.hessian import ihvp_lissa

        vector_product = 0
        model_params, _ = self.task.get_param(ckpt_idx)
        for full_data_ in self.full_train_dataloader:
            # move to device
            full_data = tuple(data.to(self.device) for data in full_data_)
            self.ihvp_func = ihvp_lissa(
                self.task.get_loss_func(),
                collate_fn=IFAttributorLiSSA.lissa_collate_fn,
                **self.transformation_kwargs,
            )
            vector_product += self.ihvp_func(
                (model_params, *full_data),
                test_rep,
                in_dims=(None,) + (0,) * len(full_data),
            ).detach()
        return vector_product

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

    def __init__(
        self,
        task: AttributionTask,
        device: Optional[str] = "cpu",
        regularization: float = 0.0,
    ) -> None:
        """Initialize the DataInf inverse Hessian attributor.

        Args:
            task (AttributionTask): The task to be attributed. Must be an instance of
                `AttributionTask`.
            device (str): Device to run the attributor on. Default is "cpu".
            regularization (float): Regularization term for Hessian vector product.
                Adding `regularization * I` to the Hessian matrix, where `I` is the
                identity matrix. Useful for singular or ill-conditioned matrices.
                Default is 0.0.
        """
        super().__init__(task, device)
        self.transformation_kwargs = {
            "regularization": regularization,
        }

    def transform_test_rep(
        self,
        ckpt_idx: int,
        full_train_rep: torch.Tensor,
        test_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the transformation on the test representations.

        Args:
            ckpt_idx (int): Index of the model checkpoints. Used for ensembling
                different trained model checkpoints.
            full_train_rep (torch.Tensor): The full training data representations.
                of shape (train_size,num_parameters)
            test_rep (torch.Tensor): Test representations to be transformed.
                Typically a 2-d tensor with shape (batch_size, num_parameters).

        Returns:
            torch.Tensor: Transformed test representations. Typically a 2-d
                tensor with shape (batch_size, transformed_dimension).
        """

        def _transform_single_test_rep(
            v: torch.Tensor,
            grad: torch.Tensor,
            regularization: float,
        ) -> torch.Tensor:
            """Transformation of a single test representation.

            DataInf assumes cross-entropy loss and approximates the Hessian
            by summation of rank-1 gradients' outer product.
            For any test representation v and training gradient grad, along
            with regularization term r:
            q = (v - (v*grad) / (r + torch.norm(grad) ** 2) * grad) / r

            This transformed test representation is later used for influence:
            Inf = dot(g,q) where g is training representation and q is transformed
            test representation.

            Args:
                v (torch.Tensor): A tensor representing (batched) test set
                representation, of shape (num_test,parameter_size)
                grad (torch.Tensor): A tensor representing a single training
                    gradient, of shape (parameter_size,)
                regularization (float): A float default to 0.1. Specifies
                    the regularization term to be added to the Hessian matrix
                    in each layer.

            Returns:
                A tensor corresponding to transformed test representation.
            """
            grad = grad.unsqueeze(-1)
            coef = (v @ grad) / (regularization + torch.norm(grad) ** 2)
            return (v - coef @ grad.T) / regularization

        regularization = self.transformation_kwargs["regularization"]
        # Split layer-wise train and test representations
        query_split = self.get_layer_wise_reps(ckpt_idx, test_rep)
        train_grad_split = self.get_layer_wise_reps(ckpt_idx, full_train_rep)
        layer_cnt = len(train_grad_split)
        ifvps = []
        # Use test batch size as intermediate batch size
        # Peak memory usage: max(train_batch_size,test_batch_size)*test_batch_size*p
        batch_size = test_rep.shape[0]
        for layer in range(layer_cnt):
            grad_layer = train_grad_split[layer]
            # length = full train data size
            length = grad_layer.shape[0]
            # Split gradients into smaller chunks
            grad_batches = grad_layer.split(batch_size, dim=0)
            running_contributions = 0
            for batch in grad_batches:
                reg = 0.1 if regularization is None else regularization
                contribution = torch.func.vmap(
                    lambda grad, layer=layer, reg=reg: _transform_single_test_rep(
                    query_split[layer],
                    grad,
                    reg,
                ),
                )(batch)
                # Sum up the contributions by chunk and average at the end
                running_contributions += contribution.sum(dim=0)
            running_contributions /= length
            ifvps.append(running_contributions)
        return torch.cat(ifvps, dim=1)

    def get_layer_wise_reps(self,
        ckpt_idx: int,
        query: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split a representation into layer-wise representations.

        Args:
            ckpt_idx (int): The index of the model parameters. This index
                is used for ensembling of different trained model.
            query (torch.Tensor): Input representations to split, could be
                train/test representations, of shape (batch_size,parameter)

        Returns:
            Tuple[torch.Tensor, ...]: The layer-wise splitted tensor, a tuple of shape
                (batch_size,layer0_size), (batch_size,layer1_size)...
        """
        model_params, param_layer_map = self.task.get_param(
            ckpt_idx, layer_split=True,
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

        for checkpoint_idx in range(len(self.task.get_checkpoints())):
            # set index to current one
            tda_output *= checkpoint_idx
            # calculate gradient with respect to full_data, do it only once
            full_train_rep = 0
            for full_data_ in self.full_train_dataloader:
                full_data = tuple(data.to(self.device).unsqueeze(0)
                             for data in full_data_)
                full_train_rep = self.generate_train_rep(
                    ckpt_idx=checkpoint_idx,
                    data=full_data,
                )

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
                        full_train_rep=full_train_rep,
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
