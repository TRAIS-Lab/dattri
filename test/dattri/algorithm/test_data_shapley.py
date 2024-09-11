"""Test for data shapley."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.data_shapley import KNNShapleyAttributor


class TestInfluenceFunction:
    """Test for data shapley."""

    def test_knn_data_shapley_exact(self):
        """Test for exact KNN data shpaley."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        test_dataset = TensorDataset(
            torch.randn(10, 1, 28, 28),
            torch.randint(0, 10, (10,)),
        )

        train_loader = DataLoader(train_dataset, batch_size=4)
        test_loader = DataLoader(test_dataset, batch_size=2)

        def f(train_batch, test_batch):
            coord1 = train_batch[0].reshape(-1, 28 * 28)
            coord2 = test_batch[0].reshape(-1, 28 * 28)
            return torch.cdist(coord1, coord2)

        attributor = KNNShapleyAttributor(k_neighbors=3, distance_func=f)
        sv = attributor.attribute(
            train_loader,
            test_loader,
            train_dataset.tensors[1],
            test_dataset.tensors[1],
        )

        # Permute train dataset and reproduce the results
        permutation = torch.randperm(train_dataset.tensors[0].size(0))
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(permutation.size(0))

        train_dataset_perm = TensorDataset(
            train_dataset.tensors[0][permutation],
            train_dataset.tensors[1][permutation],
        )
        train_loader_perm = DataLoader(train_dataset_perm, batch_size=4)
        sv_perm = attributor.attribute(
            train_loader_perm,
            test_loader,
            train_dataset_perm.tensors[1],
            test_dataset.tensors[1],
        )
        sv_perm = sv_perm[:, inverse_permutation]
        assert torch.allclose(sv, sv_perm)
