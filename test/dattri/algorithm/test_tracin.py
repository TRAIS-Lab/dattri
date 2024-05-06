"""Test for TracIn."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.tracin import TracInAttributor
from dattri.benchmark.mnist import train_mnist_lr
from dattri.func.random_projection import random_project
from dattri.func.utils import flatten_func


class TestTracInAttributor:
    """Test for TracIn."""

    def test_tracin(self):
        """Test for TracIn."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        test_dataset = TensorDataset(
            torch.randn(10, 1, 28, 28),
            torch.randint(0, 10, (10,)),
        )
        train_loader = DataLoader(train_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        model = train_mnist_lr(train_loader)

        @flatten_func(model)
        def f(params, dataloader):
            loss = nn.CrossEntropyLoss()
            loss_val = 0
            for image, label in dataloader:
                yhat = torch.func.functional_call(model, params, image)
                loss_val += loss(yhat, label)
            return loss_val

        model_params_1 = {k: p for k, p in model.named_parameters() if p.requires_grad}
        model_params_2 = {k: p for k, p in model.named_parameters() if p.requires_grad}
        params_list = [model_params_1, model_params_2]

        # train and test always share the same projector
        # checkpoints need to have differnt projectors
        proj_dim = 512
        proj_max_batch_size = 32
        rng = np.random.default_rng()
        # different seeds for each checkpoint
        seeds = rng.integers(
            low=0,
            high=500,
            size=len(params_list),
        )

        projector_list = [
            random_project(
                params_list[0],
                train_loader.batch_size,
                proj_dim,
                proj_max_batch_size,
                device="cpu",
                proj_seed=int(seed),
                use_half_precision=True,
            )
            for seed in seeds
        ]

        attributor = TracInAttributor(
            target_func=f,
            params_list=params_list,
            normalized_grad=True,
            weight_list=torch.ones(len(params_list)),
            projector_list=projector_list,
            device=torch.device("cpu"),
        )
        attributor.cache(train_loader)
        score = attributor.attribute(train_loader, test_loader)
        assert score.shape == (len(train_loader.dataset), len(test_loader.dataset))

        attributor = TracInAttributor(
            target_func=f,
            params_list=params_list,
            normalized_grad=True,
            weight_list=torch.ones(len(params_list)),
            device=torch.device("cpu"),
        )
        attributor.cache(train_loader)
        score = attributor.attribute(train_loader, test_loader)
        assert score.shape == (len(train_loader.dataset), len(test_loader.dataset))


s = TestTracInAttributor()
s.test_tracin()
