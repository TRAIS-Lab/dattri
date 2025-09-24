"""Tests for the LoGra attributor with random projection enabled."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.logra import LoGraAttributor
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

        def f(model, batch, device):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            return nn.functional.cross_entropy(outputs, targets)

        # init projectors.
        self.projector_kwargs = {
            "proj_dim": 8,
            "proj_max_batch_size": 8,
            "device": "cpu",
        }

        self.task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )
        self.attributor = LoGraAttributor(
            task=self.task,
            device="cpu",
            hessian="none",
            offload="cpu",
            projector_kwargs=self.projector_kwargs,
        )

    def test_attribute(self) -> None:
        """Ensure attribution works with non-empty projectors."""
        self.attributor.cache(self.train_loader)
        assert self.attributor.projectors, "Projectors should be initialized"
        assert self.attributor.layer_dims == [self.projector_kwargs["proj_dim"]] * len(
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

    def test_convex(self) -> None:
        """Verify repeated attribution yields identical scores."""
        score1 = self.attributor.attribute(self.train_loader, self.test_loader)
        score2 = self.attributor.attribute(self.train_loader, self.test_loader)
        assert torch.allclose(score1, score2)

    def test_matches_leave_one_out(self) -> None:
        """LoGra scores should correlate with leave-one-out retraining."""
        self.attributor.cache(self.train_loader)
        logra_score = self.attributor.attribute(self.train_loader, self.test_loader)

        criterion = nn.CrossEntropyLoss(reduction="none")

        def eval_losses(model):
            model.eval()
            losses = []
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    outputs = model(inputs)
                    losses.append(criterion(outputs, targets))
            return torch.cat(losses, dim=0)

        full_losses = eval_losses(self.task.model)
        train_ds = self.train_loader.dataset
        loo_scores = torch.zeros_like(logra_score)
        for idx in range(len(train_ds)):
            inputs = torch.cat(
                [train_ds.tensors[0][:idx], train_ds.tensors[0][idx + 1 :]],
            )
            labels = torch.cat(
                [train_ds.tensors[1][:idx], train_ds.tensors[1][idx + 1 :]],
            )
            loo_loader = DataLoader(
                TensorDataset(inputs, labels),
                batch_size=4,
                shuffle=False,
                pin_memory=False,
            )
            torch.manual_seed(0)
            model_i = train_mnist_lr(loo_loader, epoch_num=1)
            loss_i = eval_losses(model_i)
            loo_scores[idx] = full_losses - loss_i

        corr = torch.corrcoef(
            torch.stack([logra_score.flatten(), loo_scores.flatten()]),
        )[0, 1]
        assert corr > 0
