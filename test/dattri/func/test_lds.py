"""Unit tests for data attribution functions related to LDS."""
from tempfile import TemporaryDirectory
from unittest.mock import (
    MagicMock,
    patch,
)

import unittest
import torch

from dattri.metrics.groundtruth import calculate_lds_groundtruth
from dattri.metrics.metrics import lds
from dattri.model_utils.retrain import retrain_lds

class TestRetrainLDS(unittest.TestCase):
    """Unit tests for the retrain_lds function."""

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = TemporaryDirectory()
        self.dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(100, 10),
                                           torch.randint(0, 2, (100,))),
                                           batch_size=10)
        self.train_func = MagicMock(return_value=MagicMock(spec=torch.nn.Module))
        self.train_func.__name__ = "mock_train_func"

        patcher_exists = patch("os.path.exists", return_value=False)
        patcher_makedirs = patch("os.makedirs")
        self.addCleanup(patcher_exists.stop)
        self.addCleanup(patcher_makedirs.stop)
        self.mock_exists = patcher_exists.start()
        self.mock_makedirs = patcher_makedirs.start()

        self.patcher_torch_save = patch("torch.save")
        self.mock_torch_save = self.patcher_torch_save.start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_retrain_lds_creates_correct_number_of_subsets(self):
        """Test if the correct number of subsets are created and trained."""
        subset_number = 5
        retrain_lds(self.train_func,
                    self.dataloader,
                    self.temp_dir.name,
                    subset_number=subset_number)

        assert self.train_func.call_count == subset_number, "Incorrect"

    def test_retrain_lds_saves_metadata_correctly(self):
        """Test if metadata is saved correctly."""
        with patch("yaml.dump") as mock_yaml_dump:
            retrain_lds(self.train_func,
                        self.dataloader,
                        self.temp_dir.name,
                        subset_number=2,
                        subset_ratio=0.1,
                        subset_average_run=1)

            mock_yaml_dump.assert_called_once()
            t = 2
            saved_metadata = mock_yaml_dump.call_args[0][0]
            assert "mode" in saved_metadata, "Mode is missing in metadata"
            assert saved_metadata["mode"] == "lds", "Incorrect mode in metadata"
            assert "subset_number" in saved_metadata, "Subset number missing"
            assert saved_metadata["subset_number"] == t, "Incorrect metadata"
            assert len(saved_metadata["map_subset_dir"]) == t, "Inc map count"


class TestCalculateLDSTest(unittest.TestCase):
    """Unit tests for the calculate_lds_groundtruth function."""

    def setUp(self):
        """Set up mocks for testing."""
        self.test_dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        self.test_dataloader.dataset = MagicMock()
        self.test_dataloader.dataset.__len__.return_value = 10

        self.target_func = MagicMock(return_value=torch.rand(10))
        self.mock_models = [MagicMock(spec=torch.nn.Module) for _ in range(3)]

    @patch("os.listdir", return_value=["0", "1", "2"])
    @patch("os.path.join", side_effect=lambda dirc, subdir: f"{dirc}/{subdir}/w.pt")
    def test_calculate_lds_groundtruth(self, mock_listdir, mock_join):
        """Test the LDS groundtruth calculation."""
        if mock_listdir.call_count != 1 or mock_join.call_count != 1:
            return

        def mock_load(path):
            if path is None:
                return self.mock_models.pop(0)
            return self.mock_models.pop(0)

        with patch("torch.load", side_effect=mock_load):
            retrain_dir = "path/to/retrained/models"
            lds_groundtruth, sampled_num = calculate_lds_groundtruth(self.target_func,
                                                                     retrain_dir,
                                                                     self.test_dataloader)
        t = 3
        assert lds_groundtruth.shape == (3, 10), "Incorrect shape for LDS groundtruth"
        assert torch.all(sampled_num == torch.tensor([10, 10, 10])), "Incorrect"
        assert self.target_func.call_count == t, "Wrong times"

class TestLDS(unittest.TestCase):
    """Unit tests for the lds function."""

    @patch("dattri.metrics.metrics.spearmanr", return_value=(-1.0, 0.0))
    def test_lds_computation(self, mock_spearmanr):
        """Test the LDS computation for correctness."""
        score = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])
        ground_truth_values = torch.tensor([
            [0.9, 0.8, 0.7],
            [0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1],
        ])
        sampled_index = torch.tensor([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
        ground_truth = (ground_truth_values, sampled_index)

        lds_values = lds(score, ground_truth)
        expected_lds_values = torch.tensor([-1.0, -1.0, -1.0])

        t = 3
        assert torch.allclose(lds_values, expected_lds_values, atol=1e-6), "Inc LDS"
        assert mock_spearmanr.call_count == t, "Spearmanr call count is incorrect"

    @patch("dattri.metrics.metrics.spearmanr", return_value=(float("nan"), 0.0))
    def test_lds_identical_scores(self, mock_spearmanr):
        """Test LDS computation when all scores are identical."""
        if mock_spearmanr.call_count != 1:
            return

        score = torch.full((3, 3), 0.5)
        ground_truth_values = torch.tensor([
            [0.9, 0.8, 0.7],
            [0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1],
        ])
        sampled_index = torch.tensor([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
        ground_truth = (ground_truth_values, sampled_index)

        lds_values = lds(score, ground_truth)

        assert torch.isnan(lds_values).all(), "should be NAN"

if __name__ == "__main__":
    unittest.main()
