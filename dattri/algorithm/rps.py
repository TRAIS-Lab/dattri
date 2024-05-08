"""This module implement the representer point selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader

import warnings

from dattri.algorithm.utils import finetune_theta, get_rps_weight, rps_corr_check
from dattri.model_utils.hook import get_final_layer_io

from .base import BaseAttributor
from .utils import _check_shuffle


class RPSAttributor(BaseAttributor):
    """Representer point selection attributor."""

    def __init__(
        self,
        target_func: Callable,
        model: torch.nn.Module,
        final_linear_layer_name: str,
        l2_strength: float = 0.003,
        epoch: int = 3000,
        device: str = "cpu",
    ) -> None:
        """Representer point selection attributor.

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
            model (torch.nn.Module): The model to attribute. RPS will extract
                secon-to-last layer results and the final fc layer's parameter. The
                second one will be used for the initialization of the l2-finetuning.
                That is, model output = fc(second-to-last feature).
            final_linear_layer_name (str): The name of the final linear layer's name
                in the model.
            l2_strength (float): The l2 regularizaton to fine-tune the last layer.
            epoch (int): The number of epoch used to fine-tune the last layer.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.target_func = target_func
        self.model = model
        self.final_linear_layer_name = final_linear_layer_name
        self.l2_strength = l2_strength
        self.epoch = epoch
        self.device = device

    def cache(self, full_train_dataloader: DataLoader) -> None:
        """Cache the dataset for RPS calculation.

        Args:
            full_train_dataloader (DataLoader): The dataloader
                with full training samples for RPS calculation.
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

        intermediate_x_train, y_pred_train = get_final_layer_io(
            self.model,
            self.final_linear_layer_name,
            train_dataloader,
        )
        intermediate_x_test, y_pred_test = get_final_layer_io(
            self.model,
            self.final_linear_layer_name,
            test_dataloader,
        )

        # get the initial weight parameter for the final linear layer
        init_theta = getattr(self.model, self.final_linear_layer_name).weight.data
        # finetune the last layer using train samples
        best_theta = finetune_theta(
            intermediate_x_train,
            y_pred_train,
            init_theta,
            loss_func=self.target_func,
            lambda_l2=self.l2_strength,
            num_epoch=self.epoch,
        )

        # compute the RPS weight
        weight = get_rps_weight(
            best_theta,
            self.target_func,
            intermediate_x_train,
            y_pred_train,
            self.l2_strength,
        )

        # check corr for train
        rps_corr_check(weight, intermediate_x_train, y_pred_train)
        # check corr for test
        rps_corr_check(weight, intermediate_x_test, y_pred_test)

        return
