"""This file define the logistic regression model."""

import torch
from torch import nn

AVAILABLE_DATASETS = ["mnist"]


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


def create_lr_model(dataset: str, **kwargs) -> nn.Module:
    """Create a logistic regression model.

    Args:
        dataset: The dataset to create the model for.
        **kwargs: The arguments to pass to the model constructor.

    Returns:
        The logistic regression model.

    Raises:
        ValueError: If the dataset is unknown.
    """
    if dataset == "mnist":
        return LogisticRegressionMnist(**kwargs)
    message = f"Unknown dataset: {dataset}, available: {AVAILABLE_DATASETS}"
    raise ValueError(message)
