"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, List, Optional

import warnings

import torch
from torch import Tensor
from torch.func import grad
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from dattri.func.utils import flatten_params

from .base import BaseAttributor


class TracInAttributor(BaseAttributor):
    """TracIn attributor."""

    def __init__(
        self,
        target_func: Callable,
        params_list: List[dict],
        weight_list: Tensor,
        normalized_grad: bool,
        train_projector: Optional[Callable] = None,
        test_projector: Optional[Callable] = None,
        device: str = "cpu",
    ) -> None:
        """TracIn attributor initialization.

        Args:
            target_func (Callable): The target function to be attributed.
                The function can be quite flexible, but it should take the parameters
                and the dataloader as input. A typical example is as follows:
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
            params_list (List[dict]): The parameters of the target function. The key is
                the name of a parameter and the value is the parameter tensor.
                TODO: This should be changed to support a list of paths.
            weight_list (Tensor): The weight used for the "weighted sum". For
                TracIn/CosIn, this will contain a list of learning rates at each ckpt;
                for Grad-Dot/Grad-Cos, this will be a list of ones.
            normalized_grad (bool): Whether to apply normalization to gradients.
            train_projector (Callable): The projector for train gradient inner-product.
            test_projector (Callable): The projector for test gradient inner-product.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.target_func = target_func
        self.params_list = [flatten_params(params) for params in params_list]
        self.weight_list = weight_list
        self.train_projector = train_projector
        self.test_projector = test_projector
        self.normalized_grad = normalized_grad
        self.device = device
        self.full_train_dataloader = None
        self.grad_func = grad(self.target_func)

    def cache(self, full_train_dataloader: torch.utils.data.DataLoader) -> None:
        """Cache the dataset for inverse hessian calculation.

        Args:
            full_train_dataloader (torch.utils.data.DataLoader): The dataloader
                with full training samples for inverse hessian calculation.
        """
        self.full_train_dataloader = full_train_dataloader

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

        Returns:
            Tensor: The influence of the training set on the test set, with
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

        is_shuffling = isinstance(train_dataloader.sampler, RandomSampler) & isinstance(
            test_dataloader.sampler,
            RandomSampler,
        )
        if is_shuffling:
            warnings.warn(
                "The dataloader is shuffling the data. The influence \
                           calculation could not be interpreted in order.",
                stacklevel=1,
            )

        # calculate gradient of training set and testing set
        grads_train = []
        grads_test = []
        # a list of params
        for params in self.params_list:
            # for train
            param_train_grad = []
            for data in tqdm(
                train_dataloader,
                desc="calculating gradient of training set...",
                leave=False,
            ):
                loader = list(zip(*tuple(x.to(self.device).unsqueeze(0) for x in data)))
                param_train_grad.append(self.grad_func(params, loader))
            grads_train.append(torch.stack(param_train_grad, dim=0))

            # for test
            param_test_grad = []
            for data in tqdm(
                test_dataloader,
                desc="calculating gradient of testing set...",
                leave=False,
            ):
                loader = list(zip(*tuple(x.to(self.device).unsqueeze(0) for x in data)))
                param_test_grad.append(self.grad_func(params, loader))
            grads_test.append(torch.stack(param_test_grad, dim=0))

        # random projection if needed
        if self.train_projector is not None and self.test_projector is not None:
            # do normalization if needed
            if self.normalized_grad:
                weighted_score = [
                    self.train_projector(normalize(g_train))
                    @ self.test_projector(normalize(g_test)).T
                    * w
                    for g_train, g_test, w in zip(
                        grads_train,
                        grads_test,
                        self.weight_list,
                    )
                ]
            else:
                weighted_score = [
                    self.train_projector(g_train) @ self.test_projector(g_test).T * w
                    for g_train, g_test, w in zip(
                        grads_train,
                        grads_test,
                        self.weight_list,
                    )
                ]

        return torch.sum(torch.stack(weighted_score), dim=0)
