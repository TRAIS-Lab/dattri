"""This file define the MLP model."""

import torch
from torch import nn

AVAILABLE_DATASETS = ["mnist"]


class MLPMnist(nn.Module):
    """A simple MLP model for MNIST dataset."""

    def __init__(self, dropout_rate: float = 0.1) -> None:
        """Initialize the MLP model.

        Args:
            dropout_rate: The dropout rate to use.
        """
        super(MLPMnist, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x: The input image tensor.

        Returns:
            (torch.Tensor): The output tensor.
        """
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def create_mlp_model(dataset: str, **kwargs) -> nn.Module:
    """Create a MLP model.

    Args:
        dataset: The dataset to create the model for.
        **kwargs: The arguments to pass to the model constructor.

    Returns:
        The MLP model.

    Raises:
        ValueError: If the dataset is unknown.
    """
    if dataset == "mnist":
        return MLPMnist(**kwargs)
    message = f"Unknown dataset: {dataset}, available: {AVAILABLE_DATASETS}"
    raise ValueError(message)
