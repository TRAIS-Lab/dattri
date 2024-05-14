"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    from typing import Callable, List, Optional

    from torch.utils.data import DataLoader


import torch
from torch import Tensor
from torch.func import jacrev, vmap
from torch.nn.functional import normalize

from dattri.func.random_projection import random_project
from dattri.func.utils import flatten_params

from .base import BaseAttributor
from .utils import _check_shuffle


class TracInAttributor(BaseAttributor):
    """TracIn attributor."""

    def __init__(
        self,
        target_func: Callable,
        params_list: Union[List[dict]],
        weight_list: Tensor,
        normalized_grad: bool,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        """TracIn attributor initialization.

        Args:
            target_func (Callable): The target function to be attributed.
                The function can be quite flexible, but it should take the parameters
                and the (data, label) as input. A typical example is as follows:
                ```python
                @flatten_func(model)
                def f(params, image, label):
                    loss = nn.CrossEntropyLoss()
                    yhat = torch.func.functional_call(model, params, image)
                    return loss(yhat, label)
                ```.
                This examples calculates the loss of the model on input data-label pair.
            params_list (Union[List[dict], List[str]]): The parameters of the target
                function. The input should be a list of dictionaries, where the keys
                indicate the name of a parameter and the value is the parameter tensor.
            weight_list (Tensor): The weight used for the "weighted sum". For
                TracIn/CosIn, this will contain a list of learning rates at each ckpt;
                for Grad-Dot/Grad-Cos, this will be a list of ones.
            normalized_grad (bool): Whether to apply normalization to gradients.
            projector_kwargs (Optional[Dict[str, Any]]): The keyword arguments for the
                projector.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.target_func = target_func
        self.params_list = [flatten_params(params) for params in params_list]
        self.weight_list = weight_list
        # these are projector kwargs shared by train/test projector
        self.projector_kwargs = projector_kwargs
        # set proj seed
        if projector_kwargs is not None:
            self.proj_seed = self.projector_kwargs.get("proj_seed", 0)
        self.normalized_grad = normalized_grad
        self.device = device
        self.full_train_dataloader = None
        # to get per-sample gradients for a mini-batch of train/test samples
        self.grad_func = vmap(jacrev(self.target_func), in_dims=(None, 0))

    def cache(self) -> None:
        """Precompute and cache some values for efficiency."""

    def attribute(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> Tensor:
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

        Raises:
            ValueError: The length of params_list and weight_list don't match.

        Returns:
            Tensor: The influence of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).
        """
        super().attribute(train_dataloader, test_dataloader)

        _check_shuffle(train_dataloader)
        _check_shuffle(test_dataloader)

        # check the length match between params list and weight list
        if len(self.params_list) != len(self.weight_list):
            msg = "the length of params, weights lists don't match."
            raise ValueError(msg)

        # placeholder for the TDA result
        # currently assume dataloaders have .dataset method
        tda_output = torch.zeros(
            size=(len(train_dataloader.dataset), len(test_dataloader.dataset)),
            device=self.device,
        )

        # iterate over each checkpoint (each ensemble)
        for param_index, (params, params_weight) in enumerate(
            zip(self.params_list, self.weight_list),
        ):
            # prepare a checkpoint-specific seed
            if self.projector_kwargs is not None:
                ckpt_seed = self.proj_seed * int(1e5) + param_index
            for train_batch_idx, train_batch_data in enumerate(train_dataloader):
                # get gradient of train

                if self.projector_kwargs is not None:
                    # insert checkpoint-specific seed for the projector
                    self.projector_kwargs["proj_seed"] = ckpt_seed
                    # define the projector for this batch of data
                    self.train_random_project = random_project(
                        self.grad_func(params, train_batch_data),
                        # get the batch size, prevent edge case
                        train_batch_data[0].shape[0],
                        **self.projector_kwargs,
                    )

                    train_batch_grad = self.train_random_project(
                        self.grad_func(params, train_batch_data),
                    )
                else:
                    train_batch_grad = self.grad_func(params, train_batch_data)

                for test_batch_idx, test_batch_data in enumerate(test_dataloader):
                    # get gradient of test

                    if self.projector_kwargs is not None:
                        # insert checkpoint-specific seed for the projector
                        self.projector_kwargs["proj_seed"] = ckpt_seed
                        # define the projector for this batch of data
                        self.test_random_project = random_project(
                            self.grad_func(params, test_batch_data),
                            test_batch_data[0].shape[0],
                            **self.projector_kwargs,
                        )

                        test_batch_grad = self.test_random_project(
                            self.grad_func(params, test_batch_data),
                        )
                    else:
                        test_batch_grad = self.grad_func(params, test_batch_data)

                    # results position based on batch info
                    row_st = train_batch_idx * train_dataloader.batch_size
                    row_ed = min(
                        (train_batch_idx + 1) * train_dataloader.batch_size,
                        len(train_dataloader.dataset),
                    )

                    col_st = test_batch_idx * test_dataloader.batch_size
                    col_ed = min(
                        (test_batch_idx + 1) * test_dataloader.batch_size,
                        len(test_dataloader.dataset),
                    )

                    # accumulate the TDA score in corresponding positions (blocks)
                    if self.normalized_grad:
                        tda_output[row_st:row_ed, col_st:col_ed] += (
                            normalize(train_batch_grad)
                            @ normalize(test_batch_grad).T
                            * params_weight
                        )
                    else:
                        tda_output[row_st:row_ed, col_st:col_ed] += (
                            train_batch_grad @ test_batch_grad.T * params_weight
                        )

        return tda_output
