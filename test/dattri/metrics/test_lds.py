"""Unit test for lds."""

import shutil
import unittest

import torch

from dattri.metric import calculate_lds_ground_truth, lds
from dattri.model_util.retrain import retrain_lds


class TestLDSFunction(unittest.TestCase):
    """Test LDS related functions."""

    def test_lds(self):
        """Test the lds metric."""
        # num_test_samples = 2, num_train_samples = 4, num_subsets = 3, subset_size = 2
        score = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.3, 0.4]],
            dtype=torch.float32,
        )
        gt_values = torch.tensor(
            [[0.3, 0.3], [0.4, 0.4], [0.5, 0.3]],
            dtype=torch.float32,
        )
        indices = torch.tensor([[0, 1], [0, 2], [1, 2]])
        ground_truth = (gt_values, indices)

        lds_corr, lds_pval = lds(score.T, ground_truth)

        expected_corr = torch.tensor([1.0, 0.8660254037844387], dtype=torch.float32)
        expected_pval = torch.tensor([0.0, 0.3333333333333332], dtype=torch.float32)
        assert torch.allclose(
            lds_corr,
            expected_corr,
        ), "the LDS scores don't match expected values"
        assert torch.allclose(
            lds_pval,
            expected_pval,
        ), "the LDS p-values don't match expected values"

    def test_calculate_lds_ground_truth(self):
        """Test calculate_lds_ground_truth."""

        def train_func(dataloader, seed=None):  # noqa: ARG001
            if seed is not None:
                torch.manual_seed(seed)
            return torch.nn.Linear(1, 1)

        def target_func(ckpt_path, dataloader):
            model = torch.nn.Linear(1, 1)
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()
            target_values = []
            with torch.no_grad():
                for inputs, labels in dataloader:
                    outputs = model(inputs)
                    target_values.append((outputs - labels) ** 2)
            return torch.cat(target_values).squeeze()

        # num_subsets = 3, num_test_samples = 6, num_train_samples = 10

        dataset = torch.utils.data.TensorDataset(torch.randn(10, 1), torch.randn(10, 1))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        test_dataset = torch.utils.data.TensorDataset(
            torch.randn(6, 1),
            torch.randn(6, 1),
        )
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)
        path = "test_retrain_lds"

        retrain_lds(
            train_func,
            dataloader,
            path,
            num_subsets=3,
            subset_ratio=0.4,
            num_runs_per_subset=2,
        )
        gt_values, indices = calculate_lds_ground_truth(
            target_func,
            path,
            test_dataloader,
        )

        assert gt_values.shape == (
            3,
            6,
        ), f"`gt_values` has shape {gt_values.shape} but expected (3, 6)"
        assert indices.shape == (
            3,
            4,
        ), f"`indices` has shape {indices.shape} but expected (3, 4)"

        shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
