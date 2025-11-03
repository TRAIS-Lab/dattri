"""Test for TRAK."""

import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


class TestTRAK:
    """Test for TRAK."""

    def test_trak(self):
        """Test for TRAK."""
        dataset = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        model = train_mnist_lr(train_loader)

        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        def m(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return torch.exp(-loss(yhat, label_t))

        model_1 = train_mnist_lr(train_loader, epoch_num=1)
        model_2 = train_mnist_lr(train_loader, epoch_num=2)
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

    def test_trak_self_attribute(self):
        """Test for self_attribute in TRAK."""
        dataset = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        model = train_mnist_lr(train_loader)

        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        def m(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return torch.exp(-loss(yhat, label_t))

        model_1 = train_mnist_lr(train_loader, epoch_num=1)
        model_2 = train_mnist_lr(train_loader, epoch_num=2)
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

        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader).squeeze()
        assert torch.allclose(tensor1, tensor2, rtol=1e-4, atol=1e-3)

        # trak w/ cache
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
        )
        test_loader = train_loader
        tensor1 = attributor.attribute(test_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader).squeeze()
        assert torch.allclose(tensor1, tensor2, rtol=1e-4, atol=1e-3)

        # trak w/ multiple model params
        attributor = TRAKAttributor(
            task=task_m,
            correct_probability_func=m,
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
        )
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader).squeeze()
        assert torch.allclose(tensor1, tensor2, rtol=1e-4, atol=1e-3)

        shutil.rmtree(path)

    def test_trak_regularization(self):
        """Test for TRAK regularization."""
        dataset = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        model = train_mnist_lr(train_loader)

        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        def m(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return torch.exp(-loss(yhat, label_t))

        model_1 = train_mnist_lr(train_loader, epoch_num=1)
        model_2 = train_mnist_lr(train_loader, epoch_num=2)
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
            checkpoints=checkpoint_list,
        )
        # trak w/o regularization
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
        )
        score = attributor.attribute(train_loader, test_loader)
        # trak w/ regularization
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
            regularization=0.01,
        )
        score2 = attributor.attribute(train_loader, test_loader)
        assert not torch.allclose(score, score2)
        shutil.rmtree(path)
