"""Test for TRAK with dict-based data."""

import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.params.projection import TRAKProjectionParams
from dattri.task import AttributionTask

PROJ_DIM = 512
PROJ_MAX_BATCH_SIZE = 32


class TestTRAKDict:
    """Test for TRAK with dict-based data."""

    def test_trak(self):
        """Test for TRAK with dict-based data."""
        train_images = torch.randn(10, 1, 28, 28)
        train_labels = torch.randint(0, 10, (10,))

        # tuple-based loader for model training
        train_dataset_tuple = TensorDataset(train_images, train_labels)
        train_loader_tuple = DataLoader(train_dataset_tuple, batch_size=4)

        # dict-based loaders for attribution
        train_data = [
            {"image": train_images[i], "label": train_labels[i]}
            for i in range(len(train_images))
        ]
        train_loader = DataLoader(train_data, batch_size=4)
        test_loader = DataLoader(train_data, batch_size=2)

        model = train_mnist_lr(train_loader_tuple)

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

        projector_params = TRAKProjectionParams(
            proj_dim=64,
        )
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
            proj_params=projector_params,
        )
        score = attributor.attribute(train_loader, test_loader)
        score2 = attributor.attribute(train_loader, test_loader)
        assert torch.allclose(score, score2)

        # trak w/ cache
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=torch.device("cpu"),
            proj_params=projector_params,
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
            proj_params=projector_params,
        )
        attributor.cache(train_loader)
        score = attributor.attribute(test_loader)
        assert not torch.allclose(score, score2)

        shutil.rmtree(path)

    def test_trak_project_initialization_proj_dim(self):
        """Test for TRAK projection initialization with dict-based data."""
        train_images = torch.randn(10, 1, 28, 28)
        train_labels = torch.randint(0, 10, (10,))

        # tuple-based loader for model training
        train_dataset_tuple = TensorDataset(train_images, train_labels)
        train_loader_tuple = DataLoader(train_dataset_tuple, batch_size=4)

        model = train_mnist_lr(train_loader_tuple)

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

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=checkpoint_list,
        )
        # trak w/o regularization
        attributor1 = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=torch.device("cpu"),
        )

        assert attributor1.proj_params.proj_dim == PROJ_DIM
        assert attributor1.proj_params.proj_max_batch_size == PROJ_MAX_BATCH_SIZE

    def test_trak_self_attribute(self):
        """Test for self_attribute in TRAK with dict-based data."""
        train_images = torch.randn(10, 1, 28, 28)
        train_labels = torch.randint(0, 10, (10,))

        # tuple-based loader for model training
        train_dataset_tuple = TensorDataset(train_images, train_labels)
        train_loader_tuple = DataLoader(train_dataset_tuple, batch_size=10)

        # dict-based loader for attribution
        train_data = [
            {"image": train_images[i], "label": train_labels[i]}
            for i in range(len(train_images))
        ]
        train_loader = DataLoader(train_data, batch_size=10)

        model = train_mnist_lr(train_loader_tuple)

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

        projector_params = TRAKProjectionParams(
            proj_dim=64,
        )
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
            proj_params=projector_params,
        )

        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader).squeeze()
        assert torch.allclose(tensor1, tensor2, rtol=1e-4, atol=1e-3)

        # trak w/ cache
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=torch.device("cpu"),
            proj_params=projector_params,
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
            proj_params=projector_params,
        )
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader).squeeze()
        assert torch.allclose(tensor1, tensor2, rtol=1e-4, atol=1e-3)

        shutil.rmtree(path)

    def test_trak_regularization(self):
        """Test for TRAK regularization with dict-based data."""
        train_images = torch.randn(10, 1, 28, 28)
        train_labels = torch.randint(0, 10, (10,))

        # tuple-based loader for model training
        train_dataset_tuple = TensorDataset(train_images, train_labels)
        train_loader_tuple = DataLoader(train_dataset_tuple, batch_size=4)

        # dict-based loaders for attribution
        train_data = [
            {"image": train_images[i], "label": train_labels[i]}
            for i in range(len(train_images))
        ]
        train_loader = DataLoader(train_data, batch_size=4)
        test_loader = DataLoader(train_data, batch_size=2)

        model = train_mnist_lr(train_loader_tuple)

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

        projector_params = TRAKProjectionParams(
            proj_dim=64,
        )
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
            proj_params=projector_params,
        )
        score = attributor.attribute(train_loader, test_loader)
        # trak w/ regularization
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=torch.device("cpu"),
            proj_params=projector_params,
            regularization=0.01,
        )
        score2 = attributor.attribute(train_loader, test_loader)
        assert not torch.allclose(score, score2)
        shutil.rmtree(path)
