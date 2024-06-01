"""Test for influence function."""

import torch
from torch import nn
from torch.utils.data import TensorDataset

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.func.utils import flatten_func


class TestTRAK:
    """Test for influence function."""

    def test_trak(self):
        """Test for influence function."""
        dataset = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        model = train_mnist_lr(train_loader)

        @flatten_func(model)
        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        @flatten_func(model)
        def m(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return torch.exp(-loss(yhat, label_t))

        model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}

        projector_kwargs = {
            "device": "cpu",
            "use_half_precision": False,
        }

        # trak w/o cache
        attributor = TRAKAttributor(
            f,
            m,
            [model_params],
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
        )
        score = attributor.attribute(train_loader, test_loader)
        score2 = attributor.attribute(train_loader, test_loader)
        assert torch.allclose(score, score2)

        # trak w/ cache
        attributor = TRAKAttributor(
            f,
            m,
            [model_params],
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
        )
        attributor.cache(train_loader)
        score = attributor.attribute(test_loader)
        score2 = attributor.attribute(test_loader)
        assert torch.allclose(score, score2)

        # trak w/ multiple model params
        attributor = TRAKAttributor(
            f,
            m,
            [model_params, model_params],
            device=torch.device("cpu"),
            projector_kwargs=projector_kwargs,
        )
        attributor.cache(train_loader)
        score = attributor.attribute(test_loader)
        assert not torch.allclose(score, score2)
