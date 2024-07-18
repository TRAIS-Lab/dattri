"""Test mnist functions."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from dattri.benchmark.datasets.mnist import (
    LogisticRegressionMnist,
    MLPMnist,
    loss_mnist_lr,
    loss_mnist_mlp,
    train_mnist_lr,
    train_mnist_mlp,
)


class TestMnist:
    """Test mnist functions."""

    train_data = torch.randn(100, 28, 28)
    train_labels = torch.randint(0, 10, (100,))
    test_data = torch.randn(1, 28, 28)
    test_labels = torch.randint(0, 10, (1,))

    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataset = TensorDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    def test_train_mnist_lr(self):
        """Test train_mnist_lr."""
        model = train_mnist_lr(self.train_dataloader)
        assert isinstance(model, LogisticRegressionMnist)

    def test_train_mnist_mlp(self):
        """Test train_mnist_mlp."""
        model = train_mnist_mlp(self.train_dataloader)
        assert isinstance(model, MLPMnist)

    def test_loss_mnist_lr(self):
        """Test loss_mnist_lr."""
        model = train_mnist_lr(self.train_dataloader)
        torch.save(model.state_dict(), "test_model.pt")
        loss = loss_mnist_lr("test_model.pt", self.test_dataloader)
        assert isinstance(loss, torch.Tensor)

        # remove the saved model for clean up
        Path("test_model.pt").unlink(missing_ok=True)

    def test_loss_mnist_mlp(self):
        """Test loss_mnist_mlp."""
        model = train_mnist_mlp(self.train_dataloader)
        torch.save(model.state_dict(), "test_model.pt")
        loss = loss_mnist_mlp("test_model.pt", self.test_dataloader)
        assert isinstance(loss, torch.Tensor)

        # remove the saved model for clean up
        Path("test_model.pt").unlink(missing_ok=True)
