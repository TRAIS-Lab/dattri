"""This file define the logistic regression model."""

import torch
from torch import nn


class LogisticRegressionMnist(nn.Module):
    """A simple logistic regression model for MNIST dataset."""

    def __init__(self) -> None:
        """Initialize the logistic regression model."""
        super(LogisticRegressionMnist, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the logistic regression model.

        Args:
            x: The input image tensor.

        Returns:
            The output tensor.
        """
        x = x.view(x.shape[0], -1)  # Flatten the image
        return self.linear(x)
