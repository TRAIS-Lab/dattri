"""Test for influence function with dict-based data."""

# ruff: noqa: F401, PLR0914, W292

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.influence_function import (
    IFAttributorCG,
    IFAttributorExplicit,
)
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


class TestInfluenceFunctionDict:
    """Test for influence function with dict-based data."""

    def test_influence_function(self):
        """Test for influence function with dict-based data."""
        train_images = torch.randn(20, 1, 28, 28)
        train_labels = torch.randint(0, 10, (20,))
        test_images = torch.randn(10, 1, 28, 28)
        test_labels = torch.randint(0, 10, (10,))

        # tuple-based loader for model training
        train_dataset_tuple = TensorDataset(train_images, train_labels)
        train_loader_tuple = DataLoader(train_dataset_tuple, batch_size=4)

        # dict-based loaders for attribution
        train_data = [
            {"image": train_images[i], "label": train_labels[i]}
            for i in range(len(train_images))
        ]
        test_data = [
            {"image": test_images[i], "label": test_labels[i]}
            for i in range(len(test_images))
        ]
        train_loader = DataLoader(train_data, batch_size=4)
        test_loader = DataLoader(test_data, batch_size=2)

        model = train_mnist_lr(train_loader_tuple)

        def f(params, batch):
            image = batch["image"]
            label = batch["label"]
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

    def test_influence_function_self_attribute(self):
        """Test for self_attribute with dict-based data."""
        train_images = torch.randn(20, 1, 28, 28)
        train_labels = torch.randint(0, 10, (20,))

        # tuple-based loader for model training
        train_dataset_tuple = TensorDataset(train_images, train_labels)
        train_loader_tuple = DataLoader(train_dataset_tuple, batch_size=20)

        # dict-based loader for attribution
        train_data = [
            {"image": train_images[i], "label": train_labels[i]}
            for i in range(len(train_images))
        ]
        train_loader = DataLoader(train_data, batch_size=20)

        model = train_mnist_lr(train_loader_tuple)

        def f(params, batch):
            image = batch["image"]
            label = batch["label"]
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
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader)
        assert torch.allclose(tensor1, tensor2)

        # CG
        attributor = IFAttributorCG(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader)
        assert torch.allclose(tensor1, tensor2)

    def test_influence_function_partial_param(self):
        """Test for influence function partial param with dict-based data."""
        train_images = torch.randn(20, 1, 28, 28)
        train_labels = torch.randint(0, 10, (20,))
        test_images = torch.randn(10, 1, 28, 28)
        test_labels = torch.randint(0, 10, (10,))

        # tuple-based loader for model training
        train_dataset_tuple = TensorDataset(train_images, train_labels)
        train_loader_tuple = DataLoader(train_dataset_tuple, batch_size=4)

        # dict-based loaders for attribution
        train_data = [
            {"image": train_images[i], "label": train_labels[i]}
            for i in range(len(train_images))
        ]
        test_data = [
            {"image": test_images[i], "label": test_labels[i]}
            for i in range(len(test_images))
        ]
        train_loader = DataLoader(train_data, batch_size=4)
        test_loader = DataLoader(test_data, batch_size=2)

        model = train_mnist_lr(train_loader_tuple)

        def f(params, batch):
            image = batch["image"]
            label = batch["label"]
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )

        # Explicit w/ layer_name
        attributor = IFAttributorExplicit(
            task=task,
            layer_name=["linear.weight"],
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # CG
        attributor = IFAttributorCG(
            task=task,
            layer_name=["linear.weight"],
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

    def test_influence_function_ensemble(self):
        """Test for influence function ensembling with dict-based data."""
        train_images = torch.randn(20, 1, 28, 28)
        train_labels = torch.randint(0, 10, (20,))
        test_images = torch.randn(10, 1, 28, 28)
        test_labels = torch.randint(0, 10, (10,))

        # tuple-based loader for model training
        train_dataset_tuple = TensorDataset(train_images, train_labels)
        train_loader_tuple = DataLoader(train_dataset_tuple, batch_size=4)

        # dict-based loaders for attribution
        train_data = [
            {"image": train_images[i], "label": train_labels[i]}
            for i in range(len(train_images))
        ]
        test_data = [
            {"image": test_images[i], "label": test_labels[i]}
            for i in range(len(test_images))
        ]
        train_loader = DataLoader(train_data, batch_size=4)
        test_loader = DataLoader(test_data, batch_size=2)

        model = train_mnist_lr(train_loader_tuple)

        def f(params, batch):
            image = batch["image"]
            label = batch["label"]
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