"""Test utils functions."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from dattri.benchmark.utils import SubsetSampler


class TestUtils:
    """Test utils functions."""

    def test_subsetsampler(self):
        """Test subsetsampler."""
        sampler = SubsetSampler([1, 2, 3, 4, 5])
        train_data = torch.randn(100, 28, 28)
        train_labels = torch.randint(0, 10, (100,))
        train_dataset = TensorDataset(train_data, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=10, sampler=sampler)

        assert len(train_dataloader) == 1  # 5//10
