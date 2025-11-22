"""This module evaluates the performance of the data attribution."""

# ruff: noqa: ARG001, TC002
# TODO: Remove the above line after finishing the implementation of the functions.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple

import torch
from scipy.stats import pearsonr, spearmanr


def lds(
    score: torch.Tensor,
    ground_truth: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the Linear Datamodeling Score (LDS) metric.

    The LDS is calculated as the Spearman rank correlation between the predicted scores
    and the ground truth values for each test sample across all retrained models.

    Args:
        score (torch.Tensor): The data attribution score tensor with the shape
            (num_train_samples, num_test_samples).
        ground_truth (Tuple[torch.Tensor, torch.Tensor]): A tuple of two tensors. The
            first one has the shape (num_subsets, num_test_samples), which is the
            ground-truth target values for all test samples under `num_subsets` models,
            each retrained on a subset of the training data. The second tensor has the
            shape (num_subsets, subset_size), where each row refers to the indices of
            the training samples used to retrain the model.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors. The first tensor
            contains the Spearman rank correlation between the predicted scores and the
            ground truth values for each test sample. The second tensor contains the
            p-values of the correlation. Both have the shape (num_test_samples,).
    """
    score = score.T.cpu()
    gt_values, indices = ground_truth
    num_subsets = indices.shape[0]
    num_test_samples = score.shape[0]

    # Sum scores over the training subset indices
    sum_scores = torch.stack(
        [score[:, indices[i]].sum(dim=1) for i in range(num_subsets)],
        dim=0,
    )  # shape: (num_subsets, num_test_samples)

    # Calculate the Spearman rank correlation between the average scores and the
    # ground-truth values for each test sample
    lds_corr = torch.stack(
        [
            torch.tensor(
                spearmanr(sum_scores[:, i], gt_values[:, i]).correlation,
                dtype=score.dtype,
            )
            for i in range(num_test_samples)
        ],
        dim=0,
    )  # shape: (num_test_samples,)
    lds_pval = torch.stack(
        [
            torch.tensor(
                spearmanr(sum_scores[:, i], gt_values[:, i]).pvalue,
                dtype=score.dtype,
            )
            for i in range(num_test_samples)
        ],
        dim=0,
    )  # shape: (num_test_samples,)
    return lds_corr, lds_pval


def loo_corr(
    score: torch.Tensor,
    ground_truth: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Calculate the Leave-One-Out (LOO) correlation metric.

    The LOO correlation is calculated by Pearson correlation between the score
    tensor and the ground truth.

    TODO: more detailed description.

    Args:
        score (torch.Tensor): The score tensor with the shape (num_train_samples,
            num_test_samples).
        ground_truth (Tuple[torch.Tensor, torch.Tensor]): A tuple of two tensors. First
            is the LOO ground truth values for each sample in test_dataloader and each
            model in retrain_dir. The returned tensor has the shape (num_models,
            num_test_samples). Second is the tensor indicating the removed index. The
            returned tensor has the shape (num_models,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the LOO correlation
            metric values and their corresponding p-values. Both tensors have the
            shape (num_test_samples,).
    """
    score = score.cpu()
    gt_values, _ = ground_truth
    num_test_samples = score.shape[1]

    loo_corr = torch.stack(
        [
            torch.tensor(
                pearsonr(score[:, i], gt_values[:, i]).correlation,
                dtype=score.dtype,
            )
            for i in range(num_test_samples)
        ],
        dim=0,
    )  # shape: (num_test_samples,)

    loo_pval = torch.stack(
        [
            torch.tensor(
                pearsonr(score[:, i], gt_values[:, i]).pvalue,
                dtype=score.dtype,
            )
            for i in range(num_test_samples)
        ],
        dim=0,
    )  # shape: (num_test_samples,)
    return loo_corr, loo_pval


def mislabel_detection_auc(
    score: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Tuple[float, Tuple[torch.Tensor, ...]]:
    """Calculate the AUC using sorting algorithm.

    The function will calculate the false positive rates and true positive rates
    under different thresholds (number of data inspected), and return them with
    the calculated AUC (Area Under Curve).

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
    score = score.cpu()
    fpr_list, tpr_list = [0.0], [0.0]

    noise_index = set(torch.where(ground_truth)[0].numpy())
    num_noise = len(noise_index)
    num_clean = len(score) - num_noise

    # argsort the indices from low quality to high quality (scores hight to low)
    low_quality_to_high_quality = torch.argsort(score).flip(0)
    thresholds = list(range(1, len(low_quality_to_high_quality) + 1))

    for ind in thresholds:
        detected_samples = set(
            low_quality_to_high_quality[:ind].numpy(),
        ).intersection(noise_index)
        true_positive_cnt = len(detected_samples)
        false_positive_cnt = ind - true_positive_cnt

        tpr = true_positive_cnt / num_noise
        fpr = false_positive_cnt / num_clean
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    direction = 1
    tpr_list, fpr_list = torch.tensor(tpr_list), torch.tensor(fpr_list)
    auc = direction * torch.trapz(tpr_list, fpr_list)  # metrics.auc(fpr_list, tpr_list)

    # Add -np.inf to the list of thresholds, refer to sklearn.metrics.roc_curve
    thresholds = [-torch.inf, *thresholds]

    return auc, (fpr_list, tpr_list, thresholds)
