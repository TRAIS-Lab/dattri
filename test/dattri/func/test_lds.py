import unittest
import torch
from scipy.stats import spearmanr
from unittest.mock import MagicMock, patch, mock_open, Mock
from pathlib import Path


import sys
sys.path.append('/Users/jackhuang/Desktop/Jiaqi/dattri_jack')

from dattri.metrics.metrics import lds
from dattri.metrics.groundtruth import calculate_lds_groundtruth

class TestLDSFunction(unittest.TestCase):

    def test_basic_functionality(self):
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
        self.assertTrue(torch.allclose(lds_values, expected_values), "LDS values do not match expected")

    def test_basic_functionality(self):
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
        self.assertTrue(torch.allclose(lds_values, expected_values), "LDS values do not match expected")

if __name__ == '__main__':
    unittest.main()