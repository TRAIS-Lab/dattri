"""Test for TRAK with dict-based data."""

import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


class TestTRAKDict:
    """Test for TRAK with dict-based data."""

    def test_trak_dict(self):  # noqa: PLR0914
        """Test for TRAK with dict-based data."""
        images = torch.randn(10, 1, 28, 28)
        labels = torch.randint(0, 10, (10,))

        # Train model with tuple-based data (required by train_mnist_lr)
        dataset_tuple = TensorDataset(images, labels)
        train_loader_tuple = torch.utils.data.DataLoader(dataset_tuple, batch_size=4)
        model = train_mnist_lr(train_loader_tuple)

        # Create dict-based datasets from the same tensors
        data = [
            {"image": images[i], "label": labels[i]}
            for i in range(len(images))
        ]
        train_loader = torch.utils.data.DataLoader(data, batch_size=4)
        test_loader = torch.utils.data.DataLoader(data, batch_size=2)

        def f(params, batch):
            image = batch["image"]
            label = batch["label"]
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        def m(params, batch):
            image = batch["image"]
            label = batch["label"]
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return torch.exp(-loss(yhat, label_t))

        model_1 = train_mnist_lr(train_loader_tuple, epoch_num=1)
        model_2 = train_mnist_lr(train_loader_tuple, epoch_num=2)
        path = Path("./ckpts")
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(model_1.state_dict(), path / "model_1.pt")
        torch.save(model_2.state_dict(), path / "model_2.pt")

        checkpoint_list = ["ckpts/model_1.pt", "ckpts/model_2.pt"]

        projector_kwargs = {
            "device": "cpu",
        }
        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=["ckpts/model_1.pt"],
        )
        task_m = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=checkpoint_list,
        )

        # trak w/o cache
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
        )
        score = attributor.attribute(train_loader, test_loader)
        score2 = attributor.attribute(train_loader, test_loader)
        assert torch.allclose(score, score2)

        # trak w/ cache
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
        )
        attributor.cache(train_loader)
        score = attributor.attribute(test_loader)
        score2 = attributor.attribute(test_loader)
        assert torch.allclose(score, score2)

        # trak w/ multiple model params
        attributor = TRAKAttributor(
            task=task_m,
            correct_probability_func=m,
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
        )
        attributor.cache(train_loader)
        score = attributor.attribute(test_loader)
        assert not torch.allclose(score, score2)

        shutil.rmtree(path)
