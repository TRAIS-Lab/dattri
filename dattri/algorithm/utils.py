"""This module implement some utility functions for the algorithm module."""

from __future__ import annotations

import warnings
from typing import Callable, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr
from torch import Tensor, optim
from torch.autograd import Variable
from torch.func import grad



def _check_shuffle(dataloader: torch.utils.data.DataLoader) -> None:
    """Check if the dataloader is shuffling the data.

    Args:
        dataloader (torch.data.utils.DataLoader): The dataloader to be checked.
    """
    is_shuffling = isinstance(dataloader.sampler, RandomSampler)
    if is_shuffling:
        warnings.warn(
            "The dataloader is shuffling the data. The influence \
                        calculation could not be interpreted in order.",
            stacklevel=1,
        )

# The function is adapted from https://github.com/chihkuanyeh/Representer_Point_Selection/blob/master/compute_representer_vals.py
# used by RPSAttributor
def rps_finetune_model(
    x: Tensor,
    theta: torch.autograd.Variable,
) -> Tuple[Tensor, Tensor]:
    r"""To compute values related to the loss function.

    Args:
        x (Tensor): The input data.
        theta (torch.autograd.Variable): The initial last layer weight.

    Returns:
        x*\Theta^T and frobenius-norm of \Theta.
    """
    phi = torch.matmul(x, theta.transpose(0, 1))
    theta1 = torch.squeeze(theta)
    l2_norm = torch.sum(torch.mul(theta1, theta1))
    return phi, l2_norm


# The function is adapted from https://github.com/chihkuanyeh/Representer_Point_Selection/blob/master/compute_representer_vals.py
# used by RPSAttributor
def backtracking_line_search(
    theta: torch.autograd.Variable,
    grad_theta: Tensor,
    loss_func: Callable,
    x: Tensor,
    y: Tensor,
    val: float,
    lambda_l2: float,
) -> torch.autograd.Variable:
    """The backtracking line search to compute update theta.

    Args:
        theta (torch.autograd.Variable): The current last layer weight.
        grad_theta (Tensor): The current gradient of last layer weight.
        loss_func (Callable): The loss function used for prediction.
            Typically, BCELoss or CEloss.
        x (Tensor): The input data.
        y (Tensor): The pre-trained model output.
        val (float): The current loss.
        lambda_l2 (float): The l2-regularization strength.

    Returns:
        The updated last layer weight.
    """
    t = 10.0
    beta = 0.5
    min_t = 1e-10
    while True:
        cur_theta = Variable(theta - t * grad_theta, requires_grad=True)
        val_n = 0.0
        phi, l2_norm = rps_finetune_model(x, cur_theta)
        # if binary classification, output dim = 1
        val_n = loss_func(phi.float(), y.float()) + l2_norm * lambda_l2
        if t < min_t:
            return cur_theta
        armijo = val_n - val + t * (torch.norm(grad_theta) ** 2) / 2
        if armijo.data.cpu().numpy() >= 0:
            t = beta * t
        else:
            return cur_theta


# The function is adapted from https://github.com/chihkuanyeh/Representer_Point_Selection/blob/master/compute_representer_vals.py
# used by RPSAttributor
# Fine tune the last layer
def finetune_theta(
    x: Tensor,
    y: Tensor,
    init_theta: torch.autograd.Variable,
    loss_func: Callable,
    lambda_l2: float,
    num_epoch: int,
) -> Tensor:
    """To fine-tune the last layer of the model with l2-regularization.

    Args:
        x (Tensor): The input data.
        y (Tensor): The pre-trained model output.
        init_theta (torch.autograd.Variable): The initial last layer weight.
        loss_func (Callable): The loss function used for prediction.
            Typically, BCELoss or CEloss.
        lambda_l2 (float): The l2-regularization strength.
        num_epoch (int): The number of epoch used for training.

    Returns:
        The optimized last layer weight.
    """
    # activate the input y (input is raw logits)
    # depending on the number of classes
    y = torch.sigmoid(y) if init_theta.shape[0] == 1 else torch.softmax(y, dim=1)

    min_loss = 10000.0
    # define a trainable last layer
    theta = Variable(init_theta, requires_grad=True)
    optimizer = optim.SGD([theta], lr=1.0)
    for epoch in range(num_epoch):
        phi_loss = 0
        optimizer.zero_grad()
        # get the loss of the model
        phi, l2_norm = rps_finetune_model(x, theta)

        # loss func: currently either BCE or CE
        loss = l2_norm * lambda_l2 + loss_func(phi.float(), y.float())
        phi_loss += loss_func(phi.float(), y.float()).data.cpu().numpy()
        loss.backward()

        # Theta should have size (n * class_num)
        temp_theta = theta.data
        grad_loss_theta = torch.mean(torch.abs(theta.grad)).data.cpu().numpy()
        # save the Theta with lowest loss
        if grad_loss_theta < min_loss:
            if epoch == 0:
                init_grad = grad_loss_theta
            min_loss = grad_loss_theta
            best_theta = temp_theta
            if min_loss < init_grad / 200:
                break
        theta = backtracking_line_search(
            theta,
            theta.grad,
            loss_func,
            x,
            y,
            loss,
            lambda_l2,
        )
    return best_theta


# The function is adapted from https://github.com/chihkuanyeh/Representer_Point_Selection/blob/master/compute_representer_vals.py
# used by RPSAttributor
def get_rps_weight(
    best_theta: Tensor,
    loss_func: Callable,
    x_train: Tensor,
    y_train: Tensor,
    lambda_l2: float,
) -> Tensor:
    r"""Compute the decomposed RPS weight.

    Args:
        best_theta (Tensor): The optimized last layer weight.
        loss_func (Callable): The loss function used for prediction.
            Typically, BCELoss or CEloss.
        x_train (Tensor): The input feature of the training set.
        y_train (Tensor): The pre-trained model output of the training set.
        lambda_l2 (float): The l2-regularization strength.

    Returns:
        The decomposed RPS weight (\Theta^*_1 in the paper notation).
    """
    n = len(y_train)
    # caluculate theta1 based on the representer theorem's decomposition
    pre_activation_value = torch.matmul(x_train, Variable(best_theta).transpose(0, 1))
    alpha = grad(loss_func)(pre_activation_value, y_train) / (-2.0 * lambda_l2 * n)
    return torch.t(x_train) @ alpha


# The function is adapted from https://github.com/chihkuanyeh/Representer_Point_Selection/blob/master/compute_representer_vals.py
# used by RPSAttributor
# ruff: noqa: T201
def rps_corr_check(rps_weight: Tensor, x: Tensor, y: Tensor) -> None:
    """Sanity check the corr. between gt and rps prediction.

    Args:
        rps_weight (Tensor): The decomposed RPS weight.
        x (Tensor): The input feature.
        y (Tensor): The pre-trained model output.
    """
    print("--------pre activation sanity check--------")
    pre_activation_value = x @ rps_weight
    pre_y = y.data.cpu().numpy()
    y_p = pre_activation_value.data.cpu().numpy()
    print("L1 diff between gt and rps prediction per class")
    print(np.mean(np.abs(pre_y - y_p), axis=0))

    print("pearson corr between gt and rps prediction per class")
    corr_list = []
    for i in range(y.shape[1]):
        corr, _ = pearsonr(pre_y[:, i].flatten(), y_p[:, i].flatten())
        corr_list.append(corr)
    print(corr_list)

    print("--------post activation sanity check--------")
    # activate the prediction y_p (input is raw logits)
    # depending on the number of classes
    if rps_weight.shape[1] == 1:
        activated_value = torch.nn.functional.sigmoid(x @ rps_weight)
        y = torch.nn.functional.sigmoid(y)
    else:
        activated_value = torch.nn.functional.softmax(x @ rps_weight, dim=1)
        y = torch.nn.functional.softmax(y, dim=1)

    y_p = activated_value.data.cpu().numpy()
    print("L1 diff between gt and rps prediction per class")
    print(np.mean(np.abs(y.data.cpu().numpy() - y_p), axis=0))

    print("pearson corr between gt and rps prediction data per class")
    corr_list = []
    y = y.data.cpu().numpy()
    for i in range(y.shape[1]):
        corr, _ = pearsonr(y[:, i].flatten(), y_p[:, i].flatten())
        corr_list.append(corr)
    print(corr_list)