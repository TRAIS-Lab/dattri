"""Test for influence function."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.influence_function import IFAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.func.utils import flatten_func


class TestInfluenceFunction:
    """Test for influence function."""

    def test_influence_function(self):
        """Test for influence function."""
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

        model = train_mnist_lr(train_loader)

        @flatten_func(model)
        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}

        # Explicit
        attributor = IFAttributor(
            target_func=f,
            params=model_params,
            ihvp_solver="explicit",
            ihvp_kwargs={"regularization": 1e-3},
            device=torch.device("cpu"),
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # CG
        attributor = IFAttributor(
            target_func=f,
            params=model_params,
            ihvp_solver="cg",
            ihvp_kwargs={"regularization": 1e-3},
            device=torch.device("cpu"),
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # arnoldi
        attributor = IFAttributor(
            target_func=f,
            params=model_params,
            ihvp_solver="arnoldi",
            ihvp_kwargs={"regularization": 1e-3},
            device=torch.device("cpu"),
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # lissa
        attributor = IFAttributor(
            target_func=f,
            params=model_params,
            ihvp_solver="lissa",
            ihvp_kwargs={"recursion_depth": 5, "batch_size": 2},
            device=torch.device("cpu"),
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)
