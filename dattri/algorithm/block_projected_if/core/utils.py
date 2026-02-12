"""Common utility functions for gradient computation and influence attribution."""

from __future__ import annotations

import logging
from typing import Optional

import torch

# Configure logger
logger = logging.getLogger(__name__)


def stable_inverse(
    matrix: torch.Tensor,
    damping: Optional[float] = None,
) -> torch.Tensor:
    """Compute a numerically stable inverse of a matrix using eigendecomposition.

    Args:
        matrix: Input matrix to invert
        damping: (Adaptive) Damping factor for numerical stability

    Returns:
        Stable inverse of the input matrix with the same dtype as input
    """
    orig_dtype = matrix.dtype
    matrix = matrix.to(dtype=torch.float32)

    assert matrix.dim() == 2, "Input must be a 2D matrix"  # noqa: S101, PLR2004 - Development assertion for 2D matrix

    # Add damping to the diagonal
    if damping is None:
        damping = 1e-5 * torch.trace(matrix) / matrix.size(0)
    else:
        damping = damping * torch.trace(matrix) / matrix.size(0)

    damped_matrix = matrix + damping * torch.eye(matrix.size(0), device=matrix.device)

    try:
        L = torch.linalg.cholesky(damped_matrix)  # noqa: N806 - L is standard notation for lower triangular Cholesky factor
        inverse = torch.cholesky_inverse(L)
    except RuntimeError:
        logger.warning("Falling back to direct inverse due to Cholesky failure")
        inverse = torch.inverse(damped_matrix)

    return inverse.to(dtype=orig_dtype)
