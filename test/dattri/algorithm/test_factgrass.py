"""Tests for the FactGraSS attributor with two-stage projection."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm import FactGraSSAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


class TestFactGraSSAttributor:
    """Test suite for the FactGraSS attributor."""

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
        # proj_dim=64, blowup_factor=4 -> intermediate_dim=256 (16*16)
        self.attributor = FactGraSSAttributor(
            task=self.task,
            device="cpu",
            hessian="Identity",
            proj_dim=64,
            blowup_factor=4,
            offload="cpu",
        )

    def test_attribute(self) -> None:
        """Ensure attribution works with two-stage projection."""
        self.attributor.cache(self.train_loader)
        assert self.attributor.compressors, "Compressors should be initialized"
        assert self.attributor.layer_dims == [
            64,  # final proj_dim after second stage
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

    def test_invalid_dimensions(self) -> None:
        """Verify ValueError is raised when intermediate_dim is not a perfect square."""
        # proj_dim=100, blowup_factor=3 -> intermediate_dim=300 (not a perfect square)
        with pytest.raises(
            ValueError,
            match=r"intermediate_dim .* must be a perfect square",
        ):
            FactGraSSAttributor(
                task=self.task,
                device="cpu",
                hessian="Identity",
                proj_dim=100,
                blowup_factor=3,  # 100 * 3 = 300, sqrt(300) ≈ 17.32
                offload="cpu",
            )

    def test_valid_dimension_combinations(self) -> None:
        """Test that valid dimension combinations work correctly."""
        # Test case 1
        # proj_dim=4096, blowup_factor=4 -> intermediate_dim=16384 (128*128)
        attributor = FactGraSSAttributor(
            task=self.task,
            device="cpu",
            hessian="Identity",
            proj_dim=4096,
            blowup_factor=4,
            offload="cpu",
        )
        assert attributor is not None

        # Test case 2
        # proj_dim=16, blowup_factor=1 -> intermediate_dim=16 (4*4)
        attributor = FactGraSSAttributor(
            task=self.task,
            device="cpu",
            hessian="Identity",
            proj_dim=16,
            blowup_factor=1,
            offload="cpu",
        )
        assert attributor is not None

        # Test case 3
        # proj_dim=36, blowup_factor=9 -> intermediate_dim=324 (18*18)
        attributor = FactGraSSAttributor(
            task=self.task,
            device="cpu",
            hessian="Identity",
            proj_dim=36,
            blowup_factor=9,
            offload="cpu",
        )
        assert attributor is not None
