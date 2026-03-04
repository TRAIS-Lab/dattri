"""Test for influence function with dict-based data."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.influence_function import (
    IFAttributorCG,
)
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


class TestInfluenceFunctionDict:
    """Test for influence function with dict-based data."""

    def test_influence_function_dict(self):
        """Test for influence function with dict-based data."""
        train_images = torch.randn(20, 1, 28, 28)
        train_labels = torch.randint(0, 10, (20,))
        test_images = torch.randn(10, 1, 28, 28)
        test_labels = torch.randint(0, 10, (10,))

        # Train model with tuple-based data (required by train_mnist_lr)
        train_dataset_tuple = TensorDataset(train_images, train_labels)
        train_loader_tuple = DataLoader(train_dataset_tuple, batch_size=4)
        model = train_mnist_lr(train_loader_tuple)

        # Create dict-based datasets from the same tensors
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

        # CG
        attributor = IFAttributorCG(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)
