"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from dattri.func.random_projection import AbstractProjector

from functools import partial

import torch
from torch.func import grad
from tqdm import tqdm

from dattri.func.ihvp import ihvp_explicit
from dattri.func.utils import flatten_params

from .base import BaseAttributor


class IFAttributor(BaseAttributor):
    """Influence function attributor."""
    def __init__(self,
                 target_func: Callable,
                 params: dict,
                 ihvp_solver: Callable = ihvp_explicit,
                 projector: AbstractProjector = None,
                 device: str = "cpu",
                 ) -> None:
        """Influence function attributor.

        Args:
            target_func (Callable): The target function to be attributed,
                the function is quite flexible, but it should take the parameters
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
                It calculate the loss of the model on the dataloader.
            params (dict): The parameters of the target function, the key is
                the name of the parameter and the value is the parameter tensor.
                TODO: This should be changed to support a list of parameters or
                    paths for ensembling and memory efficiency.
            ihvp_solver (Callable): The solver for inverse hessian vector product
                calculation, currently we only support the non-at-x solver within the
                `dattri.func.ihvp` module.
                TODO: Make this one more flexible.
            projector (AbstractProjector): Currently this is not used.
                TODO: Enable the use of random projection for memory efficiency.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.target_func = target_func
        self.params = flatten_params(params)
        self.ihvp_solver = ihvp_solver
        self.projector = projector
        self.device = device
        self.dataloader = None
        self.grad_func = grad(self.target_func)

    def cache(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Cache the dataset for inverse hessian calculation.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader with full training
                samples for inverse hessian calculation.
        """
        self.dataloader = dataloader

    def attribute(self,
                  train_dataloader: torch.utils.data.DataLoader,
                  test_dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Calculate the influence of the training set on the test set.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for
                training samples to calculate the influence.
            test_dataloader (torch.utils.data.DataLoader): The dataloader for
                test samples to calculate the influence.

        Returns:
            torch.Tensor: The influence of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).
        """
        if self.dataloader is None:
            self.dataloader = train_dataloader

        # calculating gradient of training set
        grad_train = []
        for data in tqdm(train_dataloader,
                         desc="calculating gradient of training set..."):
            loader = zip(*tuple(x.to(self.device).unsqueeze(0) for x in data))
            grad_train.append(self.grad_func(self.params, loader))
        grad_train = torch.stack(grad_train, dim=0)

        # calculating gradient of test set
        grad_test = []
        for data in tqdm(test_dataloader,
                         desc="calculating gradient of test set..."):
            loader = zip(*tuple(x.to(self.device).unsqueeze(0) for x in data))
            grad_test.append(self.grad_func(self.params, loader))
        grad_test = torch.stack(grad_test, dim=0)

        # calculating ihvp
        ihvp = 0
        for data in tqdm(self.dataloader,
                         desc="calculating ihvp..."):
            loader = zip(*tuple(x.to(self.device).unsqueeze(0) for x in data))
            self.ihvp_func =\
                self.ihvp_solver(partial(self.target_func, dataloader=loader))
            ihvp += self.ihvp_func((self.params,), grad_test).detach()

        return grad_train @ ihvp.T
