"""Test for influence function."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.influence_function import (
    IFAttributorArnoldi,
    IFAttributorCG,
    IFAttributorDataInf,
    IFAttributorExplicit,
    IFAttributorLiSSA,
)
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


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

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )
        # Explicit
        attributor = IFAttributorExplicit(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # CG
        attributor = IFAttributorCG(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # arnoldi
        attributor = IFAttributorArnoldi(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # lissa
        attributor = IFAttributorLiSSA(
            task=task,
            device=torch.device("cpu"),
            recursion_depth=5,
            batch_size=2,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # DataInf
        # lissa
        attributor = IFAttributorDataInf(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

    def test_influence_function_ensemble(self):
        """Test for influence function with ensembling."""
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

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )
        task_m = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=[model.state_dict(), model.state_dict()],
        )
        # 1 checkpoint
        attributor = IFAttributorExplicit(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        score = attributor.attribute(train_loader, test_loader)

        # multi checkpoints
        attributor_m = IFAttributorExplicit(
            task=task_m,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor_m.cache(train_loader)
        score_m = attributor_m.attribute(train_loader, test_loader)

        assert torch.allclose(score, score_m)
