"""Test for TracIn."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.tracin import TracInAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.func.utils import flatten_func


class TestTracInAttributor:
    """Test for TracIn."""

    def test_tracin_proj(self):
        """Test for TracIn with projectors."""
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

        # the function directly operate on batches of images/labels
        # not on dataloader anymore to allow vmap usage
        # need to prepare a batch dim in the begining
        @flatten_func(model)
        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        # to simlulate multiple checkpoints
        model_params_1 = {k: p.detach() for k, p in model.named_parameters()}
        model_params_2 = {k: p.detach() for k, p in model.named_parameters()}
        params_list = [model_params_1, model_params_2]

        # train and test always share the same projector
        # checkpoints need to have differnt projectors
        pytest_device = "cpu"
        projector_kwargs = {
            "proj_dim": 512,
            "proj_max_batch_size": 32,
            "proj_seed": 42,
            "device": pytest_device,
            "use_half_precision": True,
        }

        # test with projector list
        attributor = TracInAttributor(
            target_func=f,
            params_list=params_list,
            normalized_grad=True,
            weight_list=torch.ones(len(params_list)),
            projector_kwargs=projector_kwargs,
            device=torch.device(pytest_device),
        )
        score = attributor.attribute(train_loader, test_loader)
        assert score.shape == (len(train_loader.dataset), len(test_loader.dataset))
        assert torch.count_nonzero(score) == len(train_loader.dataset) * len(
            test_loader.dataset,
        )

    def test_tracin(self):
        """Test for TracIn without projectors."""
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

        # the function directly operate on batches of images/labels
        # not on dataloader anymore to allow vmap usage
        # need to prepare a batch dim in the begining
        @flatten_func(model)
        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        # to simlulate multiple checkpoints
        model_params_1 = {k: p.detach() for k, p in model.named_parameters()}
        model_params_2 = {k: p.detach() for k, p in model.named_parameters()}
        params_list = [model_params_1, model_params_2]

        pytest_device = "cpu"
        # test with no projector list
        attributor = TracInAttributor(
            target_func=f,
            params_list=params_list,
            normalized_grad=True,
            weight_list=torch.ones(len(params_list)),
            device=torch.device(pytest_device),
        )
        score = attributor.attribute(train_loader, test_loader)
        assert score.shape == (len(train_loader.dataset), len(test_loader.dataset))
        assert torch.count_nonzero(score) == len(train_loader.dataset) * len(
            test_loader.dataset,
        )
