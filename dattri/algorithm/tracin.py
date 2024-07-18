"""This module implement TracIn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch.func import grad, vmap
from torch.nn.functional import normalize
from tqdm import tqdm

from dattri.func.projection import random_project
from dattri.func.utils import flatten_params

from .base import BaseAttributor
from .utils import _check_shuffle


class TracInAttributor(BaseAttributor):
    """TracIn attributor."""

    def __init__(
        self,
        target_func: Callable,
        model: torch.nn.Module,
        checkpoint_list: List[str],
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
            model (torch.nn.Module): The PyTorch model to be attibuted.
            checkpoint_list (List[str]): The checkpoints of the model, should be a list
                of string, indicating the stored model checkpoints' paths.
            weight_list (Tensor): The weight used for the "weighted sum". For
                TracIn/CosIn, this will contain a list of learning rates at each ckpt;
                for Grad-Dot/Grad-Cos, this will be a list of ones.
            normalized_grad (bool): Whether to apply normalization to gradients.
            projector_kwargs (Optional[Dict[str, Any]]): The keyword arguments for the
                projector.
            device (str): The device to run the attributor. Default is cpu.
        """
        self.target_func = target_func
        self.model = model
        self.checkpoint_list = checkpoint_list
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
        self.grad_func = vmap(grad(self.target_func), in_dims=(None, 0))

    def cache(self) -> None:
        """Precompute and cache some values for efficiency."""

    def attribute(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
    ) -> Tensor:
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

        Raises:
            ValueError: The length of params_list and weight_list don't match.

        Returns:
            Tensor: The influence of the training set on the test set, with
                the shape of (num_train_samples, num_test_samples).
        """
        _check_shuffle(test_dataloader)
        _check_shuffle(train_dataloader)

        # check the length match between checkpoint list and weight list
        if len(self.checkpoint_list) != len(self.weight_list):
            msg = "the length of checkpoints and weights lists don't match."
            raise ValueError(msg)

        # placeholder for the TDA result
        # should work for torch dataset without sampler
        tda_output = torch.zeros(
            size=(len(train_dataloader.sampler), len(test_dataloader.sampler)),
        )

        # iterate over each checkpoint (each ensemble)
        for ckpt_index, (ckpt, ckpt_weight) in enumerate(
            zip(self.checkpoint_list, self.weight_list),
        ):
            # load checkpoint to the model
            self.model.load_state_dict(torch.load(ckpt))
            self.model.eval()
            # get the model parameter
            parameters = {k: v.detach() for k, v in self.model.named_parameters()}

            for train_batch_idx, train_batch_data_ in enumerate(
                tqdm(
                    train_dataloader,
                    desc="calculating gradient of training set...",
                    leave=False,
                ),
            ):
                # move to device
                train_batch_data = tuple(
                    data.to(self.device) for data in train_batch_data_
                )
                # get gradient of train
                grad_t = self.grad_func(flatten_params(parameters), train_batch_data)
                if self.projector_kwargs is not None:
                    # define the projector for this batch of data
                    self.train_random_project = random_project(
                        grad_t,
                        # get the batch size, prevent edge case
                        train_batch_data[0].shape[0],
                        **self.projector_kwargs,
                    )
                    # param index as ensemble id
                    train_batch_grad = self.train_random_project(
                        torch.nan_to_num(grad_t),
                        ensemble_id=ckpt_index,
                    )
                else:
                    train_batch_grad = torch.nan_to_num(grad_t)

                for test_batch_idx, test_batch_data_ in enumerate(
                    tqdm(
                        test_dataloader,
                        desc="calculating gradient of training set...",
                        leave=False,
                    ),
                ):
                    # move to device
                    test_batch_data = tuple(
                        data.to(self.device) for data in test_batch_data_
                    )
                    # get gradient of test
                    grad_t = self.grad_func(flatten_params(parameters), test_batch_data)
                    if self.projector_kwargs is not None:
                        # define the projector for this batch of data
                        self.test_random_project = random_project(
                            grad_t,
                            test_batch_data[0].shape[0],
                            **self.projector_kwargs,
                        )

                        test_batch_grad = self.test_random_project(
                            torch.nan_to_num(grad_t),
                            ensemble_id=ckpt_index,
                        )
                    else:
                        test_batch_grad = torch.nan_to_num(grad_t)

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
                    # accumulate the TDA score in corresponding positions (blocks)
                    if self.normalized_grad:
                        tda_output[row_st:row_ed, col_st:col_ed] += (
                            (
                                normalize(train_batch_grad)
                                @ normalize(test_batch_grad).T
                                * ckpt_weight
                            )
                            .detach()
                            .cpu()
                        )
                    else:
                        tda_output[row_st:row_ed, col_st:col_ed] += (
                            (train_batch_grad @ test_batch_grad.T * ckpt_weight)
                            .detach()
                            .cpu()
                        )

        return tda_output
