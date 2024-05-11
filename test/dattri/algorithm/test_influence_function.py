"""Test for influence function."""

import torch
from torch import nn
from torch.utils.data import TensorDataset

from dattri.algorithm.influence_function import IFAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.func.utils import flatten_func


class TestInfluenceFunction:
    """Test for influence function."""

    def test_influence_function(self):
        """Test for influence function."""
        dataset = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        model = train_mnist_lr(train_loader)

        @flatten_func(model)
        def f(params, dataloader):
            loss = nn.CrossEntropyLoss()
            loss_val = 0
            for image, label in dataloader:
                yhat = torch.func.functional_call(model, params, image)
                loss_val += loss(yhat, label)
            return loss_val

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
