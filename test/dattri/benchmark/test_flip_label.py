"""Unit test for flip_label function."""

import torch

from dattri.benchmark.utils import flip_label


class TestFlipLabel:
    """Test flip_label function."""

    def test_flip_label(self):
        """Medium test for flip_label function."""
        label_range, test_size = 10, 100000
        label = torch.randint(low=0, high=label_range, size=(test_size,))
        label_copy = label.clone()

        flipped_label, noise_index = flip_label(label)
        noise_mask = torch.zeros_like(label)
        noise_mask[noise_index] = 1
        clean_mask = torch.logical_not(noise_mask)

        assert len(label) == len(flipped_label)
        assert torch.all(flipped_label[noise_index] != label[noise_index])
        assert torch.all(flipped_label[clean_mask] == label[clean_mask])

        # Check whether there is in-place modification
        assert torch.all(label == label_copy)

        # Z is the 99.99% quantile of normal distribution
        z, p = 3.719, 1 / label_range
        standard_error = torch.sqrt(torch.tensor(p * (1 - p) * test_size / label_range))

        for i in range(label_range):
            label_count = torch.count_nonzero(flipped_label[noise_index] == i)
            theoretical_mean = torch.tensor(test_size * 0.1 / label_range)
            assert torch.allclose(
                label_count.float(),
                theoretical_mean,
                atol=z * standard_error,
            )
