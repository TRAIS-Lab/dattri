"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional

import warnings
from functools import partial

import torch
from torch.func import grad
from torch.utils.data import RandomSampler
from tqdm import tqdm

from dattri.func.ihvp import ihvp_cg, ihvp_explicit
from dattri.func.utils import flatten_params

from .base import BaseAttributor

SUPPORTED_IHVP_SOLVER = {"explicit": ihvp_explicit, "cg": ihvp_cg}
SUPPORTED_PROJECTOR = {None: None}


class IFAttributor(BaseAttributor):
    """Influence function attributor."""

    def __init__(
        self,
        target_func: Callable,
        params: dict,
        ihvp_solver: str = "explicit",
        ihvp_kwargs: Optional[Dict[str, Any]] = None,
        projector: Optional[str, None] = None,  # noqa: ARG002
        projector_kwargs: Optional[Dict[str, Any]] = None,  # noqa: ARG002
        device: str = "cpu",
    ) -> None:
        """Influence function attributor.

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
            params (dict): The parameters of the target function. The key is the
                name of a parameter and the value is the parameter tensor.
                TODO: This should be changed to support a list of parameters or
                    paths for ensembling and memory efficiency.
            ihvp_solver (str): The solver for inverse hessian vector product
                calculation, currently we only support "explicit" and "cg".
            ihvp_kwargs (dict): The keyword arguments for the ihvp solver.
            projector (str): The projector for the inverse hessian vector product.
                Currently it is not supported.
            projector_kwargs (dict): The keyword arguments for the projector.
                TODO: Enable the use of random projection for memory efficiency.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.target_func = target_func
        self.params = flatten_params(params)
        if ihvp_kwargs is None:
            ihvp_kwargs = {}
        self.ihvp_solver = partial(SUPPORTED_IHVP_SOLVER[ihvp_solver], **ihvp_kwargs)
        self.projector = None
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
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Calculate the influence of the training set on the test set.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for
                training samples to calculate the influence. It can be a subset
                of the full training set if `cache` is called before. A subset
                means that only a part of the training set's influence is calculated.
                The dataloader should not be shuffled.
            test_dataloader (torch.utils.data.DataLoader): The dataloader for
                test samples to calculate the influence. The dataloader should not\
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

        # calculate gradient of training set
        grad_train = []
        for data in tqdm(
            train_dataloader,
            desc="calculating gradient of training set...",
            leave=False,
        ):
            loader = zip(*tuple(x.to(self.device).unsqueeze(0) for x in data))
            grad_train.append(self.grad_func(self.params, loader))
        grad_train = torch.stack(grad_train, dim=0)

        # calculate gradient of test set
        grad_test = []
        for data in tqdm(
            test_dataloader,
            desc="calculating gradient of test set...",
            leave=False,
        ):
            loader = zip(*tuple(x.to(self.device).unsqueeze(0) for x in data))
            grad_test.append(self.grad_func(self.params, loader))
        grad_test = torch.stack(grad_test, dim=0)

        # calculate ihvp
        ihvp = 0
        for data in tqdm(
            self.full_train_dataloader,
            desc="calculating ihvp...",
            leave=False,
        ):
            loader = zip(*tuple(x.to(self.device).unsqueeze(0) for x in data))
            self.ihvp_func = self.ihvp_solver(
                partial(self.target_func, dataloader=loader),
            )
            ihvp += self.ihvp_func((self.params,), grad_test).detach()

        return grad_train @ ihvp.T
