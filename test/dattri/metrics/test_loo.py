"""Unit test for lds."""

import unittest

import torch

from dattri.metric import loo_corr


class TestLOOFunction(unittest.TestCase):
    """Test LOO related functions."""

    def test_loo(self):
        """Test the loo_corr metric."""
        # num_test_samples = 2, num_train_samples = 4
        score = torch.tensor(
            [[0.1, 0.4], [0.2, 0.3], [0.3, 0.2], [0.4, 0.1]],
            dtype=torch.float32,
        )
        gt_values = torch.tensor(
            [[1, 1], [2, 3], [3, 2], [4, 4]],
            dtype=torch.float32,
        )
        indices = torch.tensor([0, 1, 2, 3])
        ground_truth = (gt_values, indices)

        loo_c, loo_p = loo_corr(score, ground_truth)

        expected_corr = torch.tensor([1.0, -0.8], dtype=torch.float32)
        excepted_pval = torch.tensor([4.4409e-16, 2.0000e-01], dtype=torch.float32)
        assert torch.allclose(
            loo_c,
            expected_corr,
        ), "the loo scores don't match expected values"
        assert torch.allclose(
            loo_p,
            excepted_pval,
        ), "the loo pvals don't match expected values"


if __name__ == "__main__":
    unittest.main()
