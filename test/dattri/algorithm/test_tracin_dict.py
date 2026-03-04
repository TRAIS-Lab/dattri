"""Test for TracIn with dict-based data."""

import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.tracin import TracInAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


class TestTracInAttributorDict:
    """Test for TracIn with dict-based data."""

    def test_tracin_dict(self):  # noqa: PLR0914
        """Test for TracIn without projectors with dict-based data."""
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

        # the function directly operate on batches of images/labels
        # not on dataloader anymore to allow vmap usage
        # need to prepare a batch dim in the begining
        def f(params, batch):
            image = batch["image"]
            label = batch["label"]
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        # to simlulate multiple checkpoints
        model_1 = train_mnist_lr(train_loader_tuple, epoch_num=1)
        model_2 = train_mnist_lr(train_loader_tuple, epoch_num=2)
        path = Path("./ckpts")
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(model_1.state_dict(), path / "model_1.pt")
        torch.save(model_2.state_dict(), path / "model_2.pt")

        checkpoint_list = ["ckpts/model_1.pt", "ckpts/model_2.pt"]

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=checkpoint_list,
        )

        pytest_device = "cpu"
        # test with no projector list
        attributor = TracInAttributor(
            task=task,
            weight_list=torch.ones(len(checkpoint_list)),
            normalized_grad=True,
            device=torch.device(pytest_device),
        )
        score = attributor.attribute(train_loader, test_loader)
        assert score.shape == (len(train_loader.dataset), len(test_loader.dataset))
        assert torch.count_nonzero(score) == len(train_loader.dataset) * len(
            test_loader.dataset,
        )

        shutil.rmtree(path)
