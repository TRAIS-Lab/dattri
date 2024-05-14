"""This module implement the influence function."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, List, Optional

    from torch.utils.data import DataLoader

import warnings

import torch
from torch import Tensor
from torch.func import grad
from torch.nn.functional import normalize

from dattri.func.utils import flatten_params

from .base import BaseAttributor
from .utils import _check_shuffle


class TracInAttributor(BaseAttributor):
    """TracIn attributor."""

    def __init__(
        self,
        target_func: Callable,
        params_list: List[dict],
        weight_list: Tensor,
        normalized_grad: bool,
        projector_list: Optional[List[Callable]] = None,
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
            projector_list (List[Callable]): A list of projectors used for gradient
                random projection. The length will equal len(params_list). These
                projecctors will typically have the same ProjectorType but can have
                different random seeds.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.target_func = target_func
        self.params_list = [flatten_params(params) for params in params_list]
        self.weight_list = weight_list
        self.projector_list = projector_list
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

        Raises:
            ValueError: Either two of the length of params_list, weight_list or
                project_list don't match.

        Returns:
            Tensor: The influence of the training set on the test set, with
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

        # check the length match between params, weight, and projector list
        if len(self.params_list) != len(self.weight_list):
            msg = "the length of params, weights lists don't match."
            raise ValueError(msg)

        if self.projector_list is not None and len(self.weight_list) != len(
            self.projector_list,
        ):
            msg = "the length of params, weights and projector lists don't match."
            raise ValueError(msg)

        # placeholder for the TDA result
        tda_output = torch.zeros(
            size=(len(train_dataloader), len(test_dataloader)),
            device=self.device,
        )

        # iterate over each checkpoint (each ensemble)
        for index, (params, params_weight) in enumerate(
            zip(self.params_list, self.weight_list),
        ):
            for train_batch_idx, train_batch_data in enumerate(train_dataloader):
                train_batch = list(
                    zip(
                        *tuple(
                            x.to(self.device).unsqueeze(0) for x in train_batch_data
                        ),
                    ),
                )
                # get gradient of train
                train_batch_grad = self.grad_func(params, train_batch).unsqueeze(0)
                if self.projector_list is not None:
                    train_batch_grad = self.projector_list[index](train_batch_grad)

                for test_batch_idx, test_batch_data in enumerate(test_dataloader):
                    test_batch = list(
                        zip(
                            *tuple(
                                x.to(self.device).unsqueeze(0) for x in test_batch_data
                            ),
                        ),
                    )
                    # get gradient of train
                    test_batch_grad = self.grad_func(params, test_batch).unsqueeze(0)
                    if self.projector_list is not None:
                        test_batch_grad = self.projector_list[index](test_batch_grad)

                    # results position based on batch info
                    # note that here batch_size always equal to 1
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

                    # insert the TDA score in corresponding position
                    if self.normalized_grad:
                        tda_output[row_st:row_ed, col_st:col_ed] = (
                            normalize(train_batch_grad)
                            @ normalize(test_batch_grad).T
                            * params_weight
                        )
                    else:
                        tda_output[row_st:row_ed, col_st:col_ed] = (
                            train_batch_grad @ test_batch_grad.T * params_weight
                        )

        return tda_output
