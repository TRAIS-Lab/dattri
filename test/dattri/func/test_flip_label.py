"""Unit test for flip_label function"""

import numpy as np

from dattri.datasets.utils import flip_label


class TestFlipLabel:
    """Test flip_label function"""

    def test_flip_label_medium():
        label_range, test_size = 10, 100000
        label = np.random.randint(0, label_range, size=test_size)
        label_copy = np.copy(label)

        flipped_label, noise_index = flip_label(label)

        assert len(label) == len(flipped_label)
        assert np.all(flipped_label[noise_index] != label[noise_index])
        assert np.all(flipped_label[np.logical_not(np.isin(np.arange(len(flipped_label)), noise_index))] == 
                      label[np.logical_not(np.isin(np.arange(len(label)), noise_index))])

        # Check whether there is in-place modification
        assert np.all(label == label_copy)

        # Z is the 99.99% quantile of normal distribution
        Z, p = 3.719, 1 / label_range
        standard_error = np.sqrt(p * (1 - p) * test_size / label_range)

        for i in range(label_range):
            label_count = np.count_nonzero(flipped_label[noise_index] == i)
            theoretical_mean = test_size * 0.1 / label_range
            assert np.allclose(label_count, theoretical_mean, atol=Z * standard_error), (label_count, theoretical_mean)
