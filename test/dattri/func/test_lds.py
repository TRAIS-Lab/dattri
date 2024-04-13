import unittest
import torch
import os
import numpy as np
from unittest.mock import patch, MagicMock
from tempfile import TemporaryDirectory
import sys
from scipy.stats import spearmanr
sys.path.append('/Users/jackhuang/Desktop/Jiaqi/dattri_jack')
from dattri.model_utils.retrain import retrain_lds
from dattri.metrics.groundtruth import calculate_lds_groundtruth
from dattri.metrics.metrics import lds

class TestRetrainLDS(unittest.TestCase):
    """Unit tests for the retrain_lds function."""

    def setUp(self):
        """Setup a temporary directory for testing."""
        self.temp_dir = TemporaryDirectory()
        self.dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,))), batch_size=10)
        self.train_func = MagicMock(return_value=MagicMock(spec=torch.nn.Module))
        self.train_func.__name__ = "mock_train_func"
        
        # Mocking os.path.exists and os.makedirs to prevent actual file system operations during tests
        patcher_exists = patch('os.path.exists', return_value=False)
        patcher_makedirs = patch('os.makedirs')
        self.addCleanup(patcher_exists.stop)
        self.addCleanup(patcher_makedirs.stop)
        self.mock_exists = patcher_exists.start()
        self.mock_makedirs = patcher_makedirs.start()

        # Mocking torch.save to prevent actual disk writes
        self.patcher_torch_save = patch('torch.save')
        self.mock_torch_save = self.patcher_torch_save.start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_retrain_lds_creates_correct_number_of_subsets(self):
        """Test if the correct number of subsets are created and trained."""
        subset_number = 5
        retrain_lds(self.train_func, self.dataloader, self.temp_dir.name, subset_number=subset_number)
        
        # Since subset_average_run is 1 by default, we expect the train function to be called exactly subset_number times
        self.assertEqual(self.train_func.call_count, subset_number)

    def test_retrain_lds_saves_metadata_correctly(self):
        """Test if metadata is saved correctly."""
        with patch('yaml.dump') as mock_yaml_dump:
            retrain_lds(self.train_func, self.dataloader, self.temp_dir.name, subset_number=2, subset_ratio=0.1, subset_average_run=1)
            
            # Check if yaml.dump was called once
            mock_yaml_dump.assert_called_once()

            # Extracting the saved metadata to verify its contents
            saved_metadata = mock_yaml_dump.call_args[0][0]
            self.assertIn('mode', saved_metadata)
            self.assertEqual(saved_metadata['mode'], 'lds')
            self.assertIn('subset_number', saved_metadata)
            self.assertEqual(saved_metadata['subset_number'], 2)
            self.assertIn('map_subset_dir', saved_metadata)
            self.assertEqual(len(saved_metadata['map_subset_dir']), 2)



class TestCalculateLDSTest(unittest.TestCase):
    """Unit tests for the calculate_lds_groundtruth function."""

    def setUp(self):
        """Setup mocks for testing."""
        # Mock DataLoader
        self.test_dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        self.test_dataloader.dataset = MagicMock()
        self.test_dataloader.dataset.__len__.return_value = 10  # Simulate 10 samples in the test dataset

        # Mock target function
        self.target_func = MagicMock(return_value=torch.rand(10))  # Simulate random target values for each sample

        # Create a list of mock models to simulate saved models
        self.mock_models = [MagicMock(spec=torch.nn.Module) for _ in range(3)]  # Simulate 3 models

    @patch('os.listdir', return_value=['model1.pt', 'model2.pt', 'model3.pt'])
    @patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}")
    def test_calculate_lds_groundtruth(self, mock_join, mock_listdir):
        """Test the LDS groundtruth calculation."""
        def mock_load(path):
            return self.mock_models.pop(0)
        
        with patch('torch.load', side_effect=mock_load):
            retrain_dir = 'path/to/retrained/models'
            lds_groundtruth, sampled_num = calculate_lds_groundtruth(self.target_func, retrain_dir, self.test_dataloader)

        self.assertEqual(lds_groundtruth.shape, (3, 10))  # Expect shape to be (num_models, num_test_samples)
        self.assertTrue(torch.all(sampled_num == torch.tensor([10, 10, 10])))  # Check if sampled_num is correct

        # Verify that the target function was called for each model
        self.assertEqual(self.target_func.call_count, 3)

        self.assertEqual(self.target_func.call_count, 3)

class TestLDS(unittest.TestCase):
    """Unit tests for the lds function."""

    @patch('dattri.metrics.metrics.spearmanr', return_value=(-1.0, 0.0))
    def test_lds_computation(self, mock_spearmanr):
        """Test the LDS computation for correctness."""
        # Setup test data
        score = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        ground_truth_values = torch.tensor([
            [0.9, 0.8, 0.7],
            [0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1]
        ])
        sampled_index = torch.tensor([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ])
        ground_truth = (ground_truth_values, sampled_index)

        # Compute LDS
        lds_values = lds(score, ground_truth)

        # Expected Spearman's correlation for each column
        expected_lds_values = torch.tensor([
            -1.0,  # Perfect negative correlation
            -1.0,
            -1.0
        ])

        # Verify that the lds function returns the correct Spearman rank correlations
        self.assertTrue(torch.allclose(lds_values, expected_lds_values, atol=1e-6))
        # Ensure that spearmanr was called the expected number of times (once per test sample)
        self.assertEqual(mock_spearmanr.call_count, 3)

    @patch('dattri.metrics.metrics.spearmanr', return_value=(float('nan'), 0.0))
    def test_lds_identical_scores(self, mock_spearmanr):
        """Test LDS computation when all scores are identical."""
        score = torch.full((3, 3), 0.5)
        ground_truth_values = torch.tensor([
            [0.9, 0.8, 0.7],
            [0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1]
        ])
        sampled_index = torch.tensor([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ])
        ground_truth = (ground_truth_values, sampled_index)

        # Compute LDS
        lds_values = lds(score, ground_truth)

        # Check if the results are NaN (as expected due to identical scores)
        self.assertTrue(torch.isnan(lds_values).all())

if __name__ == '__main__':
    unittest.main()
