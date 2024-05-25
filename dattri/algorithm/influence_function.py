"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple

    from torch import Tensor
    from torch.utils.data import DataLoader

import warnings
from functools import partial

import torch
from torch.func import grad, vmap
from tqdm import tqdm

from dattri.func.ihvp import ihvp_arnoldi, ihvp_cg, ihvp_explicit, ihvp_lissa
from dattri.func.utils import flatten_params

from .base import BaseAttributor
from .utils import _check_shuffle


def _lissa_collate_fn(
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
        list(
            zip(
                *tuple(
                    sampled_input[i].unsqueeze(0).float()
                    for i in range(1, len(sampled_input))
                ),
            ),
        ),
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
                calculation, currently we only support "explicit", "cg" and "arnoldi".
            ihvp_kwargs (Optional[Dict[str, Any]]): Keyword arguments for ihvp solver.
                calculation, currently we only support "explicit", "cg", "arnoldi",
                and "lissa".
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
                            grad_test_iter,
                            in_dims=(None,) + (0,) * len(loader[0]),
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
