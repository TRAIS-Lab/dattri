"""Test mnist functions."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from dattri.benchmark.mnist import (
    LogisticRegressionMnist,
    loss_mnist_lr,
    train_mnist_lr,
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
        model = train_mnist_lr(self.train_dataloader, device="cpu")
        assert isinstance(model, LogisticRegressionMnist)

    def test_loss_mnist_lr(self):
        """Test loss_mnist_lr."""
        model = train_mnist_lr(self.train_dataloader, device="cpu")
        loss = loss_mnist_lr(model, self.test_dataloader, device="cpu")
        assert isinstance(loss, float)
