"""Test mnist functions."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from dattri.benchmark.datasets.imagenet import (
    loss_imagenet_resnet18,
    train_imagenet_resnet18,
)


class TestImageNet:
    """Test mnist functions."""

    train_data = torch.randn(10, 3, 224, 224)
    train_labels = torch.randint(0, 1000, (10,))
    test_data = torch.randn(1, 3, 224, 224)
    test_labels = torch.randint(0, 1000, (1,))

    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataset = TensorDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    def test_train_imagenet_resnet18(self):
        """Test train_mnist_lr."""
        model = train_imagenet_resnet18(self.train_dataloader)
        assert isinstance(model, torch.nn.Module)

    def test_loss_imagenet_resnet18(self):
        """Test loss_imagenet_resnet18."""
        model = train_imagenet_resnet18(self.train_dataloader)
        torch.save(model.state_dict(), "test_model.pt")
        loss = loss_imagenet_resnet18("test_model.pt", self.test_dataloader)
        assert isinstance(loss, float)

        # remove the saved model for clean up
        Path("test_model.pt").unlink(missing_ok=True)
