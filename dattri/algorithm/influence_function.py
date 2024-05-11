"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional

import warnings
from functools import partial

import torch
from torch.func import grad
from tqdm import tqdm

from dattri.func.ihvp import ihvp_cg, ihvp_explicit
from dattri.func.utils import flatten_params

from .base import BaseAttributor
from .utils import _check_shuffle

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
        train_gradient_batch_size: Optional[int] = None,
        test_gradient_batch_size: Optional[int] = None,
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
            train_gradient_batch_size (int): The batch size for calculating the
                gradient of the training set. Default is None, which means the
                length of the training dataloader. The less the batch size, the
                more memory efficient, while more time consuming.
            test_gradient_batch_size (int): The batch size for calculating the
                gradient of the test set. Default is None, which means the length
                of the test dataloader. The less the batch size, the
                more memory efficient, while more time consuming.

        Returns:
            torch.Tensor: The influence of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).

        Raises:
            ValueError: If the batch size of the train and test dataloader is not 1.
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

        if train_dataloader.batch_size != 1 or test_dataloader.batch_size != 1:
            message = "The batch size of the train_dataloader\
                        and test_dataloader should be 1."
            raise ValueError(message)

        _check_shuffle(train_dataloader)
        _check_shuffle(test_dataloader)

        # TODO: the batch size should be set automatically.
        if train_gradient_batch_size is None:
            train_gradient_batch_size = len(train_dataloader)
        if test_gradient_batch_size is None:
            test_gradient_batch_size = len(test_dataloader)

        score = []
        grad_train_iter_list = []
        grad_train_counter = 0
        # TODO: sometimes the train dataloader could be swapped with the test dataloader
        for train_data in tqdm(
            train_dataloader,
            desc="calculating gradient of training set...",
            leave=False,
        ):
            # TODO: currently, vmap is not used for the gradient calculation
            train_loader = list(
                zip(*tuple(x.to(self.device).unsqueeze(0) for x in train_data)),
            )
            grad_train_iter_list.append(self.grad_func(self.params, train_loader))
            grad_train_counter += 1
            if len(
                grad_train_iter_list,
            ) < train_gradient_batch_size and grad_train_counter < len(
                train_dataloader,
            ):
                continue
            grad_train_iter = torch.stack(grad_train_iter_list, dim=0)
            grad_train_iter_list = []

            influence_train_iter_list = []
            grad_test_iter_list = []
            grad_test_counter = 0
            for test_data in tqdm(
                test_dataloader,
                desc="calculating gradient of test set...",
                leave=False,
            ):
                # TODO: currently, vmap is not used for the gradient calculation
                test_loader = list(
                    zip(*tuple(x.to(self.device).unsqueeze(0) for x in test_data)),
                )
                grad_test_iter_list.append(self.grad_func(self.params, test_loader))
                grad_test_counter += 1
                if len(
                    grad_test_iter_list,
                ) < test_gradient_batch_size and grad_test_counter < len(
                    test_dataloader,
                ):
                    continue
                grad_test_iter = torch.stack(grad_test_iter_list, dim=0)
                grad_test_iter_list = []

                ihvp = 0
                for data in tqdm(
                    self.full_train_dataloader,
                    desc="calculating ihvp...",
                    leave=False,
                ):
                    loader = list(
                        zip(*tuple(x.to(self.device).unsqueeze(0) for x in data)),
                    )
                    self.ihvp_func = self.ihvp_solver(
                        partial(self.target_func, dataloader=loader),
                    )
                    ihvp += self.ihvp_func((self.params,), grad_test_iter).detach()

                influence_train_iter_list.append(grad_train_iter @ ihvp.T)

            score.append(torch.cat(influence_train_iter_list, dim=1))

        return torch.cat(score, dim=0)
