"""Tests for the LoGra attributor with random projection enabled."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm import LoGraAttributor
from dattri.benchmark.datasets.cifar import train_cifar_resnet9
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


class TestLoGraAttributor:
    """Test suite for the LoGra attributor."""

    def setup_method(self) -> None:
        """Set up test fixtures with sample datasets and attributor configuration."""
        torch.manual_seed(0)
        train_dataset = TensorDataset(
            torch.randn(10, 1, 28, 28),
            torch.randint(0, 10, (10,)),
        )
        test_dataset = TensorDataset(
            torch.randn(4, 1, 28, 28),
            torch.randint(0, 10, (4,)),
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=False,
            pin_memory=False,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            pin_memory=False,
        )
        model = train_mnist_lr(self.train_loader, epoch_num=1)
        model.eval()

        def f(model, batch, device):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            return nn.functional.cross_entropy(outputs, targets)

        self.task = AttributionTask(
            model=model,
            loss_func=f,
            checkpoints=model.state_dict(),
        )
        self.attributor = LoGraAttributor(
            task=self.task,
            device="cpu",
            hessian="Identity",
            proj_dim=64,  # projection dimension (must be perfect square: 8*8=64)
            offload="cpu",
        )

    def test_attribute(self) -> None:
        """Ensure attribution works with non-empty projectors."""
        self.attributor.cache(self.train_loader)
        assert self.attributor.compressors, "Compressors should be initialized"
        assert self.attributor.layer_dims == [
            64,  # proj_dim = 64 (8*8)
        ] * len(
            self.attributor.layer_names,
        )
        score = self.attributor.attribute(self.train_loader, self.test_loader)
        assert score.shape == (
            len(self.train_loader.dataset),
            len(self.test_loader.dataset),
        )
        assert torch.count_nonzero(score) == len(self.train_loader.dataset) * len(
            self.test_loader.dataset,
        )

        self_inf = self.attributor.compute_self_attribution()
        assert self_inf.shape[0] == len(self.train_loader.dataset)
        assert torch.count_nonzero(self_inf) == len(self.train_loader.dataset)

    def test_identical(self) -> None:
        """Verify repeated attribution yields identical scores."""
        score1 = self.attributor.attribute(self.train_loader, self.test_loader)
        score2 = self.attributor.attribute(self.train_loader, self.test_loader)
        assert torch.allclose(score1, score2)


class TestLoGraAttributorResNet:
    """Test suite for the LoGra attributor with ResNet."""

    def setup_method(self) -> None:
        """Set up test fixtures with sample datasets and attributor configuration."""
        torch.manual_seed(0)
        train_dataset = TensorDataset(
            torch.randn(10, 3, 28, 28),
            torch.randint(0, 10, (10,)),
        )
        test_dataset = TensorDataset(
            torch.randn(4, 3, 28, 28),
            torch.randint(0, 10, (4,)),
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=False,
            pin_memory=False,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            pin_memory=False,
        )
        model = train_cifar_resnet9(self.train_loader, num_classes=10, num_epochs=1)
        model.eval()

        def f(model, batch, device):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            return nn.functional.cross_entropy(outputs, targets)

        self.task = AttributionTask(
            model=model,
            loss_func=f,
            checkpoints=model.state_dict(),
        )
        self.attributor = LoGraAttributor(
            task=self.task,
            device="cpu",
            hessian="Identity",
            proj_dim=64,  # projection dimension (must be perfect square: 8*8=64)
            offload="cpu",
        )

    def test_attribute(self) -> None:
        """Ensure attribution works with non-empty projectors."""
        # This will fail if sample_inputs[0] is used due to dimension mismatch
        self.attributor.cache(self.train_loader)
        assert self.attributor.compressors, "Compressors should be initialized"
        assert self.attributor.layer_dims == [
            64,  # proj_dim = 64 (8*8)
        ] * len(
            self.attributor.layer_names,
        )
        score = self.attributor.attribute(self.train_loader, self.test_loader)
        assert score.shape == (
            len(self.train_loader.dataset),
            len(self.test_loader.dataset),
        )
        assert torch.count_nonzero(score) == len(self.train_loader.dataset) * len(
            self.test_loader.dataset,
        )

        self_inf = self.attributor.compute_self_attribution()
        assert self_inf.shape[0] == len(self.train_loader.dataset)
        assert torch.count_nonzero(self_inf) == len(self.train_loader.dataset)
