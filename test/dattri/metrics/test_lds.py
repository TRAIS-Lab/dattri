"""Unit test for lds."""

import unittest

import torch

from dattri.metrics.groundtruth import calculate_lds_groundtruth
from dattri.metrics.metrics import lds
from dattri.model_utils.retrain import retrain_lds


class TestLDSFunction(unittest.TestCase):
    """Test LDS function."""

    def test_basic_functionality(self):
        """Test basic functionality of LDS function."""
        score = torch.tensor([[0.1, 0.2, 0.3],
                              [0.2, 0.1, 0.3],
                              [0.3, 0.2, 0.1]], dtype=torch.float32)
        gt_values = torch.tensor([[0.3, 0.3, 0.6],
                                  [0.4, 0.4, 0.4],
                                  [0.5, 0.3, 0.4]], dtype=torch.float32)
        indices = torch.tensor([[0, 1],
                                [0, 2],
                                [1, 2]])
        ground_truth = (gt_values, indices)

        lds_values = lds(score, ground_truth)

        expected_values = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        assert torch.allclose(lds_values, expected_values), "Doesn't match"

    def test_groundtruth_values(self):
        """Test groundtruth values."""

        def train_func(dataloader):  # noqa: ARG001
            return torch.nn.Linear(1, 1)

        def target_func(model_path, dataloader):
            model = torch.nn.Linear(1, 1)
            model.load_state_dict(torch.load(model_path))
            total_loss = 0
            total_samples = 0
            with torch.no_grad():
                for inputs, _ in dataloader:
                    outputs = model(inputs).item()
                    total_loss += outputs * inputs.shape[0]
                    total_samples += inputs.shape[0]
            return total_loss / total_samples

        dataset = torch.utils.data.TensorDataset(torch.randn(10, 1), torch.randn(10, 1))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        path = "test_retrain_lds"

        retrain_lds(train_func, dataloader, path, subset_number=10)
        groundtruth = calculate_lds_groundtruth(target_func, path, dataloader)

        assert groundtruth[0].shape == (10, 10), "Doesn't match"
        assert groundtruth[1].shape == (10, 5), "Doesn't match"


if __name__ == "__main__":
    unittest.main()
