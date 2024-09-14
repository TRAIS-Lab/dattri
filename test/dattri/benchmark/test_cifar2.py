"""Test CIFAR-2 functions."""

import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from dattri.benchmark.datasets.cifar import (
    create_cifar2_dataset,
    loss_cifar_resnet9,
    train_cifar_resnet9,
)


class TestCifar2:
    """Test CIFAR-2 functions."""

    train_data = torch.randn(10, 3, 32, 32)
    train_labels = torch.randint(0, 2, (10,))
    test_data = torch.randn(1, 3, 32, 32)
    test_labels = torch.randint(0, 2, (1,))

    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataset = TensorDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    def test_create_cifar2_dataset(self):
        """Test create_cifar2_dataset."""
        path = "./data"
        train_dataset, test_dataset = create_cifar2_dataset(path)
        # check length
        class_num_train = 5000
        class_num_test = 1000
        assert len(train_dataset) == class_num_train * 2
        assert len(test_dataset) == class_num_test * 2
        # check target value is 0 or 1
        for _, target in train_dataset:
            assert target in (0, 1)
        for _, target in test_dataset:
            assert target in (0, 1)

        shutil.rmtree(path)

    def test_train_cifar2_resnet9(self):
        """Test train_cifar_resnet9."""
        model = train_cifar_resnet9(self.train_dataloader, num_epochs=2)
        assert isinstance(model, torch.nn.Module)

    def test_loss_cifar2_resnet9(self):
        """Test loss_cifar2_resnet9."""
        model = train_cifar_resnet9(self.train_dataloader, num_epochs=2)
        torch.save(model.state_dict(), "test_model.pt")
        loss = loss_cifar_resnet9("test_model.pt", self.test_dataloader)
        assert isinstance(loss, torch.Tensor)

        # remove the saved model for clean up
        Path("test_model.pt").unlink(missing_ok=True)
