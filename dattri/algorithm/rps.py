"""This module implement the representer point selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    from torch import Tensor
    from torch.utils.data import DataLoader

import warnings

import torch
from torch.nn.functional import normalize

from dattri.algorithm.utils import (
    _check_shuffle,
    get_rps_weight,
    rps_finetune_theta,
)
from dattri.model_utils.hook import get_final_layer_io

from .base import BaseAttributor


class RPSAttributor(BaseAttributor):
    """Representer point selection attributor."""

    def __init__(
        self,
        loss_func: Callable,
        model: torch.nn.Module,
        final_linear_layer_name: str,
        nomralize_preactivate: bool = False,
        l2_strength: float = 0.003,
        epoch: int = 3000,
        device: str = "cpu",
    ) -> None:
        """Representer point selection attributor.

        Args:
            loss_func (Callable): The loss function to be attributed. The inputs are
                list of pre-activation values (f_i in the paper) and list of labels.
                Typical examples are BCELoss and CELoss.
            model (torch.nn.Module): The model to attribute. RPS will extract
                second-to-last layer results and the final fc layer's parameter. The
                second one will be used for the initialization of the l2-finetuning.
                That is, model output = fc(second-to-last feature).
            final_linear_layer_name (str): The name of the final linear layer's name
                in the model.
            nomralize_preactivate (bool): If set to true, then the intermediate layer
                output will be normalized. The value of the output inner-product will
                not affected by the value of individual output magnitude.
            l2_strength (float): The l2 regularizaton to fine-tune the last layer.
            epoch (int): The number of epoch used to fine-tune the last layer.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.target_func = loss_func
        self.model = model
        self.final_linear_layer_name = final_linear_layer_name
        self.nomralize_preactivate = nomralize_preactivate
        self.l2_strength = l2_strength
        self.epoch = epoch
        self.device = device
        self.full_train_dataloader = None

    def cache(self, full_train_dataloader: DataLoader) -> None:
        """Cache the full dataset for fine-tuning.

        Args:
            full_train_dataloader (DataLoader): The dataloader
                with full training samples for the last linear layer
                fine-tuning.
        """
        self.full_train_dataloader = full_train_dataloader

        # get intermediate outputs and predictions for full train dataset
        intermediate_x_train, y_pred_train = get_final_layer_io(
            self.model,
            self.final_linear_layer_name,
            self.full_train_dataloader,
            self.device,
        )
        # get the initial weight parameter for the final linear layer
        init_theta = getattr(self.model, self.final_linear_layer_name).weight.data
        # fine-tuning on the full train dataloader
        self.finetuned_theta = rps_finetune_theta(
            intermediate_x_train,
            y_pred_train,
            init_theta,
            loss_func=self.target_func,
            lambda_l2=self.l2_strength,
            num_epoch=self.epoch,
            device=self.device,
        )

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
            warnings.warn(
                "The full training data loader was NOT cached. \
                Treating the train_dataloader as the full training \
                data loader. And thus the fine-tuned last layer parameters will \
                also be based on train_dataloader",
                stacklevel=1,
            )

        _check_shuffle(train_dataloader)
        _check_shuffle(test_dataloader)

        intermediate_x_train, y_pred_train = get_final_layer_io(
            self.model,
            self.final_linear_layer_name,
            train_dataloader,
            device=self.device,
        )

        # if cache is not called before
        if self.full_train_dataloader is None:
            # get the initial weight parameter for the final linear layer
            init_theta = getattr(self.model, self.final_linear_layer_name).weight.data
            # finetune the last layer using train samples
            self.finetuned_theta = rps_finetune_theta(
                intermediate_x_train,
                y_pred_train,
                init_theta,
                loss_func=self.target_func,
                lambda_l2=self.l2_strength,
                num_epoch=self.epoch,
                device=self.device,
            )

        # get intermediate features for test samples
        intermediate_x_test, _ = get_final_layer_io(
            self.model,
            self.final_linear_layer_name,
            test_dataloader,
            device=self.device,
        )

        y_test = torch.cat([target for _, target in test_dataloader], dim=0)

        alpha = get_rps_weight(
            self.finetuned_theta,
            self.target_func,
            intermediate_x_train,
            y_pred_train,
            y_test,
            self.l2_strength,
        )

        if self.nomralize_preactivate:
            return (
                normalize(intermediate_x_train)
                @ normalize(intermediate_x_test).T
                * alpha
            )

        return intermediate_x_train @ intermediate_x_test.T * alpha
