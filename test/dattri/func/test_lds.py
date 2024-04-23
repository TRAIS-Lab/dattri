"""Unit test for lds."""

import sys
import unittest

import torch

sys.path.append("/Users/jackhuang/Desktop/Jiaqi/dattri_jack")

from dattri.metrics.metrics import lds


class TestLDSFunction(unittest.TestCase):
    """Test LDS function."""

    def test_basic_functionality(self):
        """Test basic functionality of LDS function."""
        score = torch.tensor([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.6],
                              [0.7, 0.8, 0.9]], dtype=torch.float32)
        gt_values = torch.tensor([[10, 30, 50],
                                  [20, 60, 80],
                                  [7, 8, 9]], dtype=torch.float32)
        indices = torch.tensor([[0, 1, 2],
                                [0, 1, 2],
                                [0, 1, 2]])
        ground_truth = (gt_values, indices)

        lds_values = lds(score, ground_truth)

        expected_values = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        assert torch.allclose(lds_values, expected_values), "Doesn't match."

    def test_basic_functionality_diff_indices(self):
        """Test basic functionality of LDS function."""
        score = torch.tensor([[0.1, 0.2, 0.3],
                              [0.4, 0.5, 0.6],
                              [0.7, 0.8, 0.9]], dtype=torch.float32)
        gt_values = torch.tensor([[10, 30],
                                  [20, 60],
                                  [7, 8]], dtype=torch.float32)
        indices = torch.tensor([[0, 1],
                                [0, 1],
                                [0, 1]])
        ground_truth = (gt_values, indices)

        lds_values = lds(score, ground_truth)

        expected_values = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        assert torch.allclose(lds_values, expected_values), "Doesn't match"


if __name__ == "__main__":
    unittest.main()
