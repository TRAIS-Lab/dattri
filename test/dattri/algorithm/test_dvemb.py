"""Tests for the Data Value Embedding (DVEmb)."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.dvemb import DVEmbAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.func.utils import flatten_func


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
        self.device = "cpu"

        @flatten_func(self.model)
        def per_sample_loss_fn(params, data_tensors):
            image, label = data_tensors
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
            yhat = torch.func.functional_call(self.model, params, image)
            return nn.CrossEntropyLoss()(yhat, label)

        self.loss_func = per_sample_loss_fn

    def _run_dvemb_simulation(self, attributor: DVEmbAttributor):
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

        attributor.compute_embeddings()
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
            model=self.model,
            loss_func=self.loss_func,
            device=self.device,
        )
        self._run_dvemb_simulation(attributor)

    def test_dvemb_with_projection(self):
        """Test DVEmb with random projection."""
        projection_dim = 16
        attributor = DVEmbAttributor(
            model=self.model,
            loss_func=self.loss_func,
            device=self.device,
            projection_dim=projection_dim,
        )
        self._run_dvemb_simulation(attributor)
        assert attributor.projector is not None
        assert attributor.cached_gradients[0][0].shape[1] == projection_dim


if __name__ == "__main__":
    pytest.main([__file__])
