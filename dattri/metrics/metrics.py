"""This module evaluate the performance of the data attribution."""

# ruff: noqa: ARG001, TCH002
# TODO: Remove the above line after finishing the implementation of the functions.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple

import sklearn
import torch


def lds(score: torch.Tensor,
        ground_truth: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Calculate the Linear Datamodeling Score (LDS) metric.

    TODO: Add the LDS metric description.

    Args:
        score (torch.Tensor): The score tensor with the shape
            (num_train_samples, num_test_samples).
        ground_truth (torch.Tensor): A tuple of two tensors. First is the LDS
            groundtruth values for each sample in test_dataloader and each model
            in retrain_dir. The returned tensor has the shape
            (num_models, num_test_samples). Second is the tensor indicating the
            sampled index. The returned tensor has the shape
            (num_models, sampled_num).

    Returns:
        torch.Tensor: The LDS metric value. The returned tensor has the shape
            (num_test_samples,).
    """
    return None


def loo_corr(score: torch.Tensor,
             ground_truth: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Calculate the Leave-One-Out (LOO) correlation metric.

    The LOO correlation is calculated by pearson correlation between the score
    tensor and the groundtruth.

    TODO: more detailed description.

    Args:
        score (torch.Tensor): The score tensor with the shape (num_train_samples,
            num_test_samples).
        ground_truth (torch.Tensor): A tuple of two tensors. First is the LOO
            groundtruth values for each sample in test_dataloader and each model
            in retrain_dir. The returned tensor has the shape (num_models,
            num_test_samples). Second is the tensor indicating the removed index. The
            returned tensor has the shape (num_models,).

    Returns:
        torch.Tensor: The LOO correlation metric value. The returned tensor has the
            shape (num_test_samples,).
    """
    return None


def mislabel_detection_auc(score: torch.Tensor,
                           ground_truth: torch.Tensor,
                           ) -> Tuple[float, Tuple[torch.Tensor, ...]]:
    """Calculate the AUC using sorting algorithm.

    The function will calculate the false positive rates and true positive rates
    under different thresholds (number of data inspected), and return them with
    the calculated auc (Area Under Curve).

    Args:
        score (torch.Tensor): The self-attribution scores of shape (num_train_samples,).
        ground_truth (torch.Tensor): A tensor indicating the noise index.
            The returned binary tensor has the shape (num_train_samples,).

    Returns:
        (Tuple[float, Tuple[float, ...]]): A tuple with 2 items.
        The first is the AUROC value (float),
        the second is a Tuple with `fpr, tpr, thresholds` just like
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html.
    """
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(ground_truth, score)
    fpr, tpr = torch.tensor(fpr), torch.tensor(tpr)
    auc = sklearn.metrics.auc(fpr, tpr)

    return auc, (fpr, tpr, thresholds)
