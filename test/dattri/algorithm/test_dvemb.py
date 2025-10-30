"""Tests for the Data Value Embedding (DVEmb)."""

import math

import pytest
import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.dvemb import DVEmbAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


class TestDVEmbAttributor:
    """Test suite for the DVEmb attributor."""

    def setup_method(self):
        """Set up the test environment."""
        torch.manual_seed(0)
        self.train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        self.test_dataset = TensorDataset(
            torch.randn(10, 1, 28, 28),
            torch.randint(0, 10, (10,)),
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=2, shuffle=False)

        self.model = train_mnist_lr(self.train_loader, epoch_num=1)
        self.criterion = nn.CrossEntropyLoss()

        def functional_loss(params, data):
            images, labels = data
            y_hat = functional_call(self.model, params, (images,))
            return self.criterion(y_hat, labels.unsqueeze(0))

        self.task = AttributionTask(
            model=self.model,
            loss_func=functional_loss,
            checkpoints=[self.model.state_dict()],
        )

    def _run_dvemb_simulation(self, attributor: DVEmbAttributor):
        """A generic simulation runner for any configured DVEmbAttributor."""
        num_epochs = 2
        learning_rate = 0.01
        for epoch in range(num_epochs):
            for i, (data, target) in enumerate(self.train_loader):
                start_index = i * self.train_loader.batch_size
                indices = torch.arange(start_index, start_index + len(data))
                attributor.cache_gradients(
                    epoch,
                    (data, target),
                    indices,
                    learning_rate,
                )

        attributor.compute_embeddings(clear_cache=False)
        for epoch_embedding in attributor.embeddings.values():
            assert not torch.isnan(epoch_embedding).any()

        total_score = attributor.attribute(self.test_loader)
        assert total_score.shape == (
            len(self.train_dataset),
            len(self.test_dataset),
        )
        assert not torch.isnan(total_score).any()

        epoch_0_score = attributor.attribute(self.test_loader, epoch=0)
        epoch_1_score = attributor.attribute(self.test_loader, epoch=1)
        assert epoch_0_score.shape == (
            len(self.train_dataset),
            len(self.test_dataset),
        )
        assert torch.allclose(
            total_score,
            epoch_0_score + epoch_1_score,
            rtol=1e-3,
            atol=1e-2,
        )

        subset_indices = [1, 2]
        subset_score = attributor.attribute(
            self.test_loader,
            traindata_indices=subset_indices,
        )
        assert subset_score.shape == (len(subset_indices), len(self.test_dataset))
        assert torch.allclose(subset_score, total_score[subset_indices, :])

    def test_dvemb_no_projection(self):
        """Test DVEmb without random projection."""
        attributor = DVEmbAttributor(
            task=self.task,
            criterion=self.criterion,
            factorization_type="none",
        )
        self._run_dvemb_simulation(attributor)

    def test_dvemb_with_projection(self):
        """Test DVEmb with random projection."""
        proj_dim = 16
        attributor = DVEmbAttributor(
            task=self.task,
            criterion=self.criterion,
            proj_dim=proj_dim,
            factorization_type="none",
        )
        self._run_dvemb_simulation(attributor)
        assert attributor.projector is not None
        assert attributor.cached_gradients[0][0].shape[1] == proj_dim

    def test_dvemb_kronecker_no_projection(self):
        """Test DVEmb with Kronecker factorization and no projection."""
        attributor = DVEmbAttributor(
            task=self.task,
            criterion=self.criterion,
            factorization_type="kronecker",
        )
        self._run_dvemb_simulation(attributor)
        assert attributor.cached_factors

    def test_dvemb_kronecker_with_projection(self):
        """Test DVEmb with Kronecker factorization and projection."""
        proj_dim = 36
        attributor = DVEmbAttributor(
            task=self.task,
            criterion=self.criterion,
            factorization_type="kronecker",
            proj_dim=proj_dim,
        )
        self._run_dvemb_simulation(attributor)
        assert attributor.use_factorization
        assert attributor.random_projectors is not None

        num_layers = len(attributor._linear_layers)
        expected_proj_dim = int(math.sqrt(proj_dim / num_layers))
        assert attributor.projection_dim == expected_proj_dim

        assert (
            attributor.cached_factors[0][0][0]["A"].shape[1]
            == attributor.projection_dim
        )

    def test_dvemb_elementwise_with_projection(self):
        """Test DVEmb with elementwise factorization and projection."""
        proj_dim = 16
        attributor = DVEmbAttributor(
            task=self.task,
            criterion=self.criterion,
            factorization_type="elementwise",
            proj_dim=proj_dim,
        )
        self._run_dvemb_simulation(attributor)
        assert attributor.use_factorization
        assert attributor.random_projectors is not None
        grad_dim = attributor.cached_gradients[0][0].shape[1]
        assert grad_dim == attributor.projection_dim

    def test_dvemb_kronecker_with_layer_names(self):
        """Test DVEmb with Kronecker factorization and specified layer names."""
        target_layer = "linear"
        attributor = DVEmbAttributor(
            task=self.task,
            criterion=self.criterion,
            factorization_type="kronecker",
            layer_names=[target_layer],
        )
        assert len(attributor._linear_layers) == 1
        assert (
            attributor._linear_layers[0]
            == dict(self.model.named_modules())[target_layer]
        )
        self._run_dvemb_simulation(attributor)
        assert len(attributor.cached_factors[0][0]) == 1


if __name__ == "__main__":
    pytest.main([__file__])
