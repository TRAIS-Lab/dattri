"""This module implements the TracIn attributor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import List, Literal, Optional, Union

    from dattri.task import AttributionTask

import torch
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm

from dattri.algorithm.block_projected_if.offload import create_offload_manager
from dattri.func.projection import random_project

from .base import BaseAttributor
from .utils import _check_shuffle

DEFAULT_PROJECTOR_KWARGS = {
    "proj_dim": 512,
    "proj_max_batch_size": 32,
    "proj_seed": 0,
    "device": "cpu",
}


class TracInAttributor(BaseAttributor):
    """TracIn attributor."""

    def __init__(
        self,
        task: AttributionTask,
        weight_list: Tensor,
        normalized_grad: bool,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        device: str = "cpu",
        offload: Literal["none", "cpu", "disk"] = "none",
        cache_dir: Optional[str] = None,
        chunk_size: int = 16,
    ) -> None:
        """Initialize the TracIn attributor.

        Args:
            task (AttributionTask): The task to be attributed. Please refer to the
                `AttributionTask` for more details.
            weight_list (Tensor): The weight used for the "weighted sum". For
                TracIn/CosIn, this will contain a list of learning rates at each ckpt;
                for Grad-Dot/Grad-Cos, this will be a list of ones.
            normalized_grad (bool): Whether to apply normalization to gradients.
            projector_kwargs (Optional[Dict[str, Any]]): The keyword arguments for the
                projector.
            layer_name (Optional[Union[str, List[str]]]): The name of the layer to be
                used to calculate the train/test representations. If None, full
                parameters are used. This should be a string or a list of strings
                if multiple layers are needed. The name of layer should follow the
                key of model.named_parameters(). Default: None.
            device (str): The device to run the attributor. Default is cpu.
            offload: Memory management strategy ("none", "cpu", "disk"), stating
                the place to offload the gradients.
                "cpu": stores gradients on CPU and moves to device when needed.
                "disk": stores gradients on disk and moves to device when needed.
            cache_dir: Directory for caching (required when offload="disk").
            chunk_size: Chunk size for processing in disk offload.
        """
        self.task = task
        self.weight_list = weight_list
        # these are projector kwargs shared by train/test projector
        self.projector_kwargs = DEFAULT_PROJECTOR_KWARGS
        if projector_kwargs is not None:
            self.projector_kwargs.update(projector_kwargs)
        self.normalized_grad = normalized_grad
        self.layer_name = layer_name
        self.device = device
        self.offload = offload
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.full_train_dataloader = None
        self._offload_managers = []
        # to get per-sample gradients for a mini-batch of train/test samples
        self.grad_target_func = self.task.get_grad_target_func(in_dims=(None, 0))
        self.grad_loss_func = self.task.get_grad_loss_func(in_dims=(None, 0))

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """Cache the dataset for gradient calculation.

        Args:
            full_train_dataloader (torch.utils.data.DataLoader): The dataloader
                with full training samples for gradient calculation.

        Raises:
            ValueError: If the length of checkpoints and weight list don't match.
        """
        _check_shuffle(full_train_dataloader)
        self.full_train_dataloader = full_train_dataloader

        # check the length match between checkpoint list and weight list
        if len(self.task.get_checkpoints()) != len(self.weight_list):
            msg = "the length of checkpoints and weights lists don't match."
            raise ValueError(msg)

        # Initialize offload managers (one per checkpoint)
        self._offload_managers = []
        layer_names = ["grad"]  # Dummy layer name for API compatibility
        for _ in range(len(self.task.get_checkpoints())):
            offloader = create_offload_manager(
                offload_type=self.offload,
                device=self.device,
                layer_names=layer_names,
                cache_dir=self.cache_dir,
                chunk_size=self.chunk_size,
            )
            self._offload_managers.append(offloader)

        for ckpt_idx in range(len(self.task.get_checkpoints())):
            parameters, _ = self.task.get_param(
                ckpt_idx=ckpt_idx,
                layer_name=self.layer_name,
            )

            if self.layer_name is not None:
                self.grad_target_func = self.task.get_grad_target_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )
                self.grad_loss_func = self.task.get_grad_loss_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )

            for batch_idx, train_batch_data_ in enumerate(
                tqdm(
                    full_train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
                ),
            ):
                # move to device
                if isinstance(train_batch_data_, (tuple, list)):
                    train_batch_data = tuple(
                        data.to(self.device) for data in train_batch_data_
                    )
                else:
                    train_batch_data = train_batch_data_
                grad_t = self.grad_loss_func(parameters, train_batch_data)
                if self.projector_kwargs is not None:
                    # define the projector for this batch of data
                    self.train_random_project = random_project(
                        grad_t,
                        train_batch_data[0].shape[0],
                        **self.projector_kwargs,
                    )
                    # param index as ensemble id
                    train_batch_grad = self.train_random_project(
                        torch.nan_to_num(grad_t),
                        ensemble_id=ckpt_idx,
                    )
                else:
                    train_batch_grad = torch.nan_to_num(grad_t)
                # Store using offload manager (wrap as list for API compatibility)
                self._offload_managers[ckpt_idx].store_gradients(
                    batch_idx,
                    [train_batch_grad.clone().detach()],
                    is_test=False,
                )

    def attribute(  # noqa: PLR0912, PLR0915
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tensor:
        """Calculate the influence of the training set on the test set.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for
                training samples to calculate the influence. It can be a subset
                of the full training set if `cache` is called before. A subset
                means that only a part of the training set's influence is calculated.
                The dataloader should not be shuffled.
            test_dataloader (torch.utils.data.DataLoader): The dataloader for
                test samples to calculate the influence. The dataloader should not
                be shuffled.

        Returns:
            Tensor: The influence of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).

        Raises:
            ValueError: The length of params_list and weight_list don't match.
            ValueError: If the train_dataloader is not None and the full training
                dataloader is cached or no train_loader is provided in both cases.
        """
        _check_shuffle(test_dataloader)
        if train_dataloader is not None:
            _check_shuffle(train_dataloader)

        if train_dataloader is not None and self.full_train_dataloader is not None:
            msg = "You have cached a training loader by .cache()\
                       and you are trying to attribute a different training loader.\
                       If this new training loader is a subset of the cached training\
                       loader, please don't input the training dataloader in\
                       .attribute() and directly use index to select the corresponding\
                       scores."
            raise ValueError(msg)
        if train_dataloader is None and self.full_train_dataloader is None:
            msg = "You did not state a training loader in .attribute() and you\
                       did not cache a training loader by .cache(). Please provide a\
                       training loader or cache a training loader."
            raise ValueError(msg)
        # check the length match between checkpoint list and weight list
        if len(self.task.get_checkpoints()) != len(self.weight_list):
            msg = "the length of checkpoints and weights lists don't match."
            raise ValueError(msg)

        # placeholder for the TDA result
        # should work for torch dataset without sampler
        tda_output = torch.zeros(
            size=(
                len((train_dataloader or self.full_train_dataloader).sampler),
                len(test_dataloader.sampler),
            ),
        )
        # use normalize or identity depending on config,
        norm = normalize if self.normalized_grad else lambda x: x

        # iterate over each checkpoint (each ensemble)
        for ckpt_idx, ckpt_weight in zip(
            range(len(self.task.get_checkpoints())),
            self.weight_list,
        ):
            parameters, _ = self.task.get_param(
                ckpt_idx=ckpt_idx,
                layer_name=self.layer_name,
            )

            if self.layer_name is not None:
                self.grad_target_func = self.task.get_grad_target_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )
                self.grad_loss_func = self.task.get_grad_loss_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )

            if train_dataloader is not None:
                for train_batch_idx, train_batch_data_ in enumerate(
                    tqdm(
                        train_dataloader,
                        desc="calculating gradient of training set...",
                        leave=False,
                    ),
                ):
                    # move to device
                    if isinstance(train_batch_data_, (tuple, list)):
                        train_batch_data = tuple(
                            x.to(self.device) for x in train_batch_data_
                        )
                    else:
                        train_batch_data = train_batch_data_
                    grad_t = self.grad_loss_func(parameters, train_batch_data)
                    if self.projector_kwargs is not None:
                        # define the projector for this batch of data
                        self.train_random_project = random_project(
                            grad_t,
                            # get the batch size, prevent edge case
                            train_batch_data[0].shape[0],
                            **self.projector_kwargs,
                        )
                        # param index as ensemble id
                        train_batch_grad = self.train_random_project(
                            torch.nan_to_num(grad_t),
                            ensemble_id=ckpt_idx,
                        )
                    else:
                        train_batch_grad = torch.nan_to_num(grad_t)

                    for test_batch_idx, test_batch_data_ in enumerate(
                        tqdm(
                            test_dataloader,
                            desc="calculating gradient of test set...",
                            leave=False,
                        ),
                    ):
                        # move to device
                        if isinstance(test_batch_data_, (tuple, list)):
                            test_batch_data = tuple(
                                x.to(self.device) for x in test_batch_data_
                            )
                        else:
                            test_batch_data = test_batch_data_
                        grad_t = self.grad_target_func(parameters, test_batch_data)
                        if self.projector_kwargs is not None:
                            # define the projector for this batch of data
                            self.test_random_project = random_project(
                                grad_t,
                                test_batch_data[0].shape[0],
                                **self.projector_kwargs,
                            )

                            test_batch_grad = self.test_random_project(
                                torch.nan_to_num(grad_t),
                                ensemble_id=ckpt_idx,
                            )
                        else:
                            test_batch_grad = torch.nan_to_num(grad_t)

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
                        # accumulate the TDA score in corresponding positions (blocks)
                        tda_output[row_st:row_ed, col_st:col_ed] += (
                            (
                                norm(train_batch_grad)
                                @ norm(test_batch_grad).T
                                * ckpt_weight
                            )
                            .detach()
                            .cpu()
                        )

            else:
                # For "none" mode: concat all cached grads into one tensor
                # for a single efficient matmul per test batch
                if self.offload == "none":
                    all_train_grads = torch.cat(
                        [
                            self._offload_managers[ckpt_idx].retrieve_gradients(
                                i,
                                is_test=False,
                            )[0]
                            for i in range(len(self.full_train_dataloader))
                        ],
                        dim=0,
                    )
                    all_train_grads = norm(all_train_grads)

                for test_batch_idx, test_batch_data_ in enumerate(
                    tqdm(
                        test_dataloader,
                        desc="calculating gradient of test set...",
                        leave=False,
                    ),
                ):
                    # move to device
                    if isinstance(test_batch_data_, (tuple, list)):
                        test_batch_data = tuple(
                            x.to(self.device) for x in test_batch_data_
                        )
                    else:
                        test_batch_data = test_batch_data_
                    grad_t = self.grad_target_func(parameters, test_batch_data)
                    if self.projector_kwargs is not None:
                        # define the projector for this batch of data
                        self.test_random_project = random_project(
                            grad_t,
                            test_batch_data[0].shape[0],
                            **self.projector_kwargs,
                        )
                        test_batch_grad = self.test_random_project(
                            torch.nan_to_num(grad_t),
                            ensemble_id=ckpt_idx,
                        )
                    else:
                        test_batch_grad = torch.nan_to_num(grad_t)

                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.sampler),
                    )

                    if self.offload == "none":
                        # all train grads are already in memory and pre-normalized above
                        tda_output[:, col_st:col_ed] += (
                            (all_train_grads @ norm(test_batch_grad).T * ckpt_weight)
                            .detach()
                            .cpu()
                        )
                    else:
                        # For cpu/disk offload: retrieve train grads batch by batch
                        # to keep memory footprint low
                        for train_batch_idx in range(len(self.full_train_dataloader)):
                            train_batch_grad = self._offload_managers[
                                ckpt_idx
                            ].retrieve_gradients(
                                train_batch_idx,
                                is_test=False,
                            )[0]

                            row_st = (
                                train_batch_idx * self.full_train_dataloader.batch_size
                            )
                            row_ed = min(
                                (train_batch_idx + 1)
                                * self.full_train_dataloader.batch_size,
                                len(self.full_train_dataloader.sampler),
                            )

                            tda_output[row_st:row_ed, col_st:col_ed] += (
                                (
                                    norm(train_batch_grad)
                                    @ norm(test_batch_grad).T
                                    * ckpt_weight
                                )
                                .detach()
                                .cpu()
                            )

        return tda_output

    def self_attribute(
        self,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> Tensor:
        """Calculate the influence of the training set on itself.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for
                training samples to calculate the influence. It can be a subset
                of the full training set if `cache` is called before. A subset
                means that only a part of the training set's influence is calculated.
                The dataloader should not be shuffled.

        Returns:
            Tensor: The influence of the training set on itself, with
                the shape of (num_train_samples,).

        Raises:
            ValueError: The length of params_list and weight_list don't match.
        """
        test_dataloader = train_dataloader
        _check_shuffle(test_dataloader)
        _check_shuffle(train_dataloader)

        # check the length match between checkpoint list and weight list
        if len(self.task.get_checkpoints()) != len(self.weight_list):
            msg = "the length of checkpoints and weights lists don't match."
            raise ValueError(msg)

        # placeholder for the TDA result
        # should work for torch dataset without sampler
        tda_output = torch.zeros(
            size=(len(train_dataloader.sampler),),
        )

        # iterate over each checkpoint (each ensemble)
        for ckpt_idx, ckpt_weight in zip(
            range(len(self.task.get_checkpoints())),
            self.weight_list,
        ):
            parameters, _ = self.task.get_param(
                ckpt_idx=ckpt_idx,
                layer_name=self.layer_name,
            )
            if self.layer_name is not None:
                self.grad_target_func = self.task.get_grad_target_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
                )
                self.grad_loss_func = self.task.get_grad_loss_func(
                    in_dims=(None, 0),
                    layer_name=self.layer_name,
                    ckpt_idx=ckpt_idx,
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
                    data.to(self.device) for data in train_batch_data_
                )
                grad_t = self.grad_loss_func(parameters, train_batch_data)
                if self.projector_kwargs is not None:
                    # define the projector for this batch of data
                    self.train_random_project = random_project(
                        grad_t,
                        # get the batch size, prevent edge case
                        train_batch_data[0].shape[0],
                        **self.projector_kwargs,
                    )
                    # param index as ensemble id
                    train_batch_grad = self.train_random_project(
                        torch.nan_to_num(grad_t),
                        ensemble_id=ckpt_idx,
                    )
                else:
                    train_batch_grad = torch.nan_to_num(grad_t)
                row_st = train_batch_idx * train_dataloader.batch_size
                row_ed = min(
                    (train_batch_idx + 1) * train_dataloader.batch_size,
                    len(train_dataloader.sampler),
                )
                if self.normalized_grad:
                    tda_output[row_st:row_ed] += (
                        (torch.ones(row_ed - row_st) * ckpt_weight).detach().cpu()
                    )
                else:
                    tda_output[row_st:row_ed] += (
                        (
                            torch.einsum("ij,ij->i", train_batch_grad, train_batch_grad)
                            * ckpt_weight
                        )
                        .detach()
                        .cpu()
                    )

        return tda_output
