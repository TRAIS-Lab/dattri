"""This module implement the TRAK."""

# ruff: noqa: N806

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional

    from dattri.task import AttributionTask


import torch
from torch.func import vmap
from tqdm import tqdm

from dattri.func.projection import random_project
from dattri.func.utils import _unflatten_params

from .base import BaseAttributor
from .utils import _check_shuffle

DEFAULT_PROJECTOR_KWARGS = {
    "proj_dim": 512,
    "proj_max_batch_size": 32,
    "proj_seed": 0,
    "device": "cpu",
    "use_half_precision": False,
}


class TRAKAttributor(BaseAttributor):
    """TRAK attributor."""

    def __init__(
        self,
        task: AttributionTask,
        correct_probability_func: Callable,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the TRAK attributor.

        Args:
            task (AttributionTask): The task to be attributed. Please refer to the
                `AttributionTask` for more details.
            correct_probability_func (Callable): The function to calculate the
                probability to correctly predict the label of the input data.
                A typical example is as follows:
                ```python
                @flatten_func(model)
                def m(params, image_label_pair):
                    image, label = image_label_pair
                    image_t = image.unsqueeze(0)
                    label_t = label.unsqueeze(0)
                    loss = nn.CrossEntropyLoss()
                    yhat = torch.func.functional_call(model, params, image_t)
                    p = torch.exp(-loss(yhat, label_t))
                    return p
                ```.
            projector_kwargs (Optional[Dict[str, Any]], optional): The kwargs for the
                random projection. Defaults to None.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.task = task
        self.norm_scaler = (
            sum(p.numel() for p in self.task.get_param(index=0)[0]) ** 0.5
        )
        self.projector_kwargs = DEFAULT_PROJECTOR_KWARGS
        if projector_kwargs is not None:
            self.projector_kwargs.update(projector_kwargs)
        self.device = device
        self.grad_target_func = self.task.get_grad_target_func(in_dims=(None, 0))
        self.grad_loss_func = self.task.get_grad_loss_func(in_dims=(None, 0))
        self.correct_probability_func = vmap(
            correct_probability_func,
            in_dims=(None, 0),
        )
        self.full_train_dataloader = None

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """Cache the dataset for gradient calculation.

        Args:
            full_train_dataloader (torch.utils.data.DataLoader): The dataloader
                with full training samples for gradient calculation.
        """
        _check_shuffle(full_train_dataloader)
        self.full_train_dataloader = full_train_dataloader
        inv_XTX_XT_list = []
        running_Q = 0
        running_count = 0
        for ckpt_seed in range(len(self.task.get_checkpoints())):
            parameters, _ = self.task.get_param(index=ckpt_seed)

            full_train_projected_grad = []
            Q = []
            for train_data in tqdm(
                self.full_train_dataloader,
                desc="calculating gradient of training set...",
                leave=False,
            ):
                train_batch_data = tuple(data.to(self.device) for data in train_data)
                grad_t = self.grad_loss_func(parameters, train_batch_data)
                grad_t = torch.nan_to_num(grad_t)
                grad_t /= self.norm_scaler
                grad_p = (
                    random_project(
                        grad_t,
                        train_batch_data[0].shape[0],
                        **self.projector_kwargs,
                    )(grad_t, ensemble_id=ckpt_seed)
                    .clone()
                    .detach()
                )
                full_train_projected_grad.append(grad_p)
                Q.append(
                    (
                        torch.ones(train_batch_data[0].shape[0]).to(self.device)
                        - self.correct_probability_func(
                            _unflatten_params(parameters, self.task.get_model()),
                            train_batch_data,
                        ).flatten()
                    )
                    .clone()
                    .detach(),
                )
            full_train_projected_grad = torch.cat(full_train_projected_grad, dim=0)
            Q = torch.cat(Q, dim=0)
            inv_XTX_XT = (
                torch.linalg.inv(
                    full_train_projected_grad.T @ full_train_projected_grad,
                )
                @ full_train_projected_grad.T
            )
            inv_XTX_XT_list.append(inv_XTX_XT)
            running_Q = running_Q * running_count + Q
            running_count += 1  # noqa: SIM113
            running_Q /= running_count
        self.inv_XTX_XT_list = inv_XTX_XT_list
        self.Q = running_Q

    def attribute(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
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

        Raises:
            ValueError: If the train_dataloader is not None and the full training
                dataloader is cached or no train_loader is provided in both cases.
        """
        _check_shuffle(test_dataloader)
        if train_dataloader is not None:
            _check_shuffle(train_dataloader)

        running_xinv_XTX_XT = 0
        running_Q = 0
        running_count = 0
        if train_dataloader is not None and self.full_train_dataloader is not None:
            message = "You have cached a training loader by .cache()\
                       and you are trying to attribute a different training loader.\
                       If this new training loader is a subset of the cached training\
                       loader, please don't input the training dataloader in\
                       .attribute() and directly use index to select the corresponding\
                       scores."
            raise ValueError(message)
        if train_dataloader is None and self.full_train_dataloader is None:
            message = "You did not state a training loader in .attribute() and you\
                       did not cache a training loader by .cache(). Please provide a\
                       training loader or cache a training loader."
            raise ValueError(message)
        for ckpt_seed in range(len(self.task.get_checkpoints())):
            parameters, _ = self.task.get_param(index=ckpt_seed)

            if train_dataloader is not None:
                train_projected_grad = []
                Q = []
                for train_data in tqdm(
                    train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
                ):
                    train_batch_data = tuple(
                        data.to(self.device) for data in train_data
                    )
                    grad_t = self.grad_loss_func(
                        parameters,
                        train_batch_data,
                    )
                    grad_t = torch.nan_to_num(grad_t)
                    grad_t /= self.norm_scaler

                    grad_p = (
                        random_project(
                            grad_t,
                            train_batch_data[0].shape[0],
                            **self.projector_kwargs,
                        )(grad_t, ensemble_id=ckpt_seed)
                        .clone()
                        .detach()
                    )
                    train_projected_grad.append(grad_p)
                    Q.append(
                        (
                            torch.ones(train_batch_data[0].shape[0]).to(self.device)
                            - self.correct_probability_func(
                                _unflatten_params(parameters, self.task.get_model()),
                                train_batch_data,
                            )
                        )
                        .clone()
                        .detach(),
                    )
                train_projected_grad = torch.cat(train_projected_grad, dim=0)
                Q = torch.cat(Q, dim=0)

            test_projected_grad = []
            for test_data in tqdm(
                test_dataloader,
                desc="calculating gradient of test set...",
                leave=False,
            ):
                test_batch_data = tuple(data.to(self.device) for data in test_data)
                grad_t = self.grad_target_func(parameters, test_batch_data)
                grad_t = torch.nan_to_num(grad_t)
                grad_t /= self.norm_scaler

                grad_p = (
                    random_project(
                        grad_t,
                        test_batch_data[0].shape[0],
                        **self.projector_kwargs,
                    )(grad_t, ensemble_id=ckpt_seed)
                    .clone()
                    .detach()
                )
                test_projected_grad.append(grad_p)
            test_projected_grad = torch.cat(test_projected_grad, dim=0)

            if train_dataloader is not None:
                running_xinv_XTX_XT = (
                    running_xinv_XTX_XT * running_count
                    + test_projected_grad
                    @ torch.linalg.inv(train_projected_grad.T @ train_projected_grad)
                    @ train_projected_grad.T
                )
            else:
                running_xinv_XTX_XT = (
                    running_xinv_XTX_XT * running_count
                    + test_projected_grad @ self.inv_XTX_XT_list[ckpt_seed]
                )

            if train_dataloader is not None:
                running_Q = running_Q * running_count + Q
            running_count += 1  # noqa: SIM113
            if train_dataloader is not None:
                running_Q /= running_count
            running_xinv_XTX_XT /= running_count

        if train_dataloader is not None:
            return (running_xinv_XTX_XT @ running_Q.diag().to(self.device)).T
        return (running_xinv_XTX_XT @ self.Q.diag().to(self.device)).T
