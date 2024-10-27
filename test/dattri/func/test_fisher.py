"""Unit test for ifvp calculator."""


import numpy as np
import torch
from torch import nn, optim

from dattri.func.fisher import (
    ifvp_at_x_explicit,
    ifvp_explicit,
)
from dattri.func.hessian import ihvp_at_x_explicit
from dattri.func.utils import flatten_func, flatten_params


class TestIFVP:
    """Test ifvp functions."""

    def test_ifvp_explicit(self):  # noqa: PLR0914
        """Test ifvp_explicit."""
        # Generate synthetic dataset
        num_samples_per_class = 500
        input_dim = 2
        rng = np.random.default_rng()

        class0_data = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(num_samples_per_class, input_dim),
        ) + np.array([2, 2])
        class1_data = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(num_samples_per_class, input_dim),
        ) + np.array([-2, -2])

        data = np.vstack((class0_data, class1_data))
        labels = np.hstack(
            (np.zeros(num_samples_per_class), np.ones(num_samples_per_class)),
        )
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Hyperparameters
        hidden_dim = 10
        output_dim = 2
        num_epochs = 1000

        # Model, loss function, and optimizer
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Training loop
        for _ in range(num_epochs):
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        @flatten_func(model, param_num=0)
        def ce_y(params, data):
            yhat = torch.func.functional_call(model, params, data[0])
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(yhat, data[1])

        @flatten_func(model, param_num=0)
        def ce_yhat(params, data):
            yhat = torch.func.functional_call(model, params, data[0])
            loss_fn = torch.nn.CrossEntropyLoss()
            label = torch.argmax(yhat, dim=1)
            return loss_fn(yhat, label)

        model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}

        ifvp_explicit_func = ifvp_explicit(ce_yhat, argnums=0, regularization=0.1)
        ifvp_at_x_explicit_func = ifvp_at_x_explicit(
            ce_yhat,
            flatten_params(model_params),
            (data, labels),
            argnums=0,
            regularization=0.1,
        )
        ihvp_explicit_func = ihvp_at_x_explicit(
            ce_y,
            flatten_params(model_params),
            (data, labels),
            argnums=0,
            regularization=0.1,
        )

        v = torch.randn(1, 52)

        def corr(tensor1, tensor2):
            mean1 = torch.mean(tensor1)
            mean2 = torch.mean(tensor2)
            covariance = torch.mean((tensor1 - mean1) * (tensor2 - mean2))
            variance1 = torch.mean((tensor1 - mean1) ** 2)
            variance2 = torch.mean((tensor2 - mean2) ** 2)
            return covariance / (torch.sqrt(variance1) * torch.sqrt(variance2))

        ifvp = ifvp_explicit_func((flatten_params(model_params), (data, labels)), v)
        ifvp_at_x = ifvp_at_x_explicit_func(v)
        ihvp = ihvp_explicit_func(v)

        tol = 0.9
        assert corr(ifvp, ihvp) > tol
        assert corr(ifvp_at_x, ihvp) > tol
