"""Unit test for ifvp calculator."""

import types

import numpy as np
import torch
from torch import nn, optim

from dattri.func.fisher import (
    EKFAC_CACHE_KEY,
    MLPCache,
    ifvp_at_x_datainf,
    ifvp_at_x_ekfac,
    ifvp_at_x_explicit,
    ifvp_datainf,
    ifvp_explicit,
    manual_cache_forward,
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

    def test_ifvp_ekfac(self):
        """Test ifvp_ekfac_at_x."""
        dim_in, dim_out = 100, 1
        sample_size = 5000

        class LinearModel(torch.nn.Module):
            def __init__(self) -> None:
                super(LinearModel, self).__init__()
                self.linear = torch.nn.Linear(dim_in, dim_out, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.view(x.shape[0], -1)
                return self.linear(x)

        @manual_cache_forward
        def custom_forward_method(self, hidden_states):
            if not hasattr(self, EKFAC_CACHE_KEY):
                # Normal forward pass
                hidden_states = hidden_states.view(hidden_states.shape[0], -1)
                return self.linear(hidden_states)

            # Forward pass with caching i/o variables
            cache = getattr(self, EKFAC_CACHE_KEY)
            x1 = hidden_states.view(hidden_states.shape[0], -1)
            y1 = self.linear(x1)
            cache.input_hidden_pairs.append((x1, y1))
            return y1

        model = LinearModel()
        model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}

        x = torch.randn(sample_size, dim_in)
        y = torch.randn(sample_size, dim_out)
        v = torch.randn(5, dim_out, dim_in)

        @flatten_func(model, param_num=0)
        def f(params):
            logits = torch.func.functional_call(model, params, x)
            return torch.mean((logits - y) ** 2)

        @flatten_func(model, param_num=0)
        def f_ekfac(params, x, y):
            # The custom function should zero out gradient
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

            logits = torch.func.functional_call(model, params, x)
            return torch.mean((logits - y) ** 2, dim=-1)

        ifvp_explicit_at_x_func = ihvp_at_x_explicit(
            f,
            flatten_params(model_params),
            argnums=0,
        )

        ground_truth = ifvp_explicit_at_x_func(v.reshape(5, -1))

        model.forward = types.MethodType(custom_forward_method, model)
        cache = MLPCache()
        setattr(model, EKFAC_CACHE_KEY, cache)

        ifvp_at_x_ekfac_func = ifvp_at_x_ekfac(
            f_ekfac,
            *(flatten_params(model_params), x, y),
            in_dims=(None, 0, 0),
            mlp_cache=cache,
            batch_size=128,
            damping=0.1,
        )

        estimation = ifvp_at_x_ekfac_func([[v]])[0][0]
        for i in range(5):
            corr = np.corrcoef(
                ground_truth[i].detach().numpy(),
                estimation[i].detach().numpy(),
            )[0, 1]
            assert corr > 0.95  # noqa: PLR2004

    def test_ifvp_ekfac_conv(self):
        """Test ifvp_ekfac_at_x with conv."""
        dim_in, dim_out = 100, 1
        kernel_size = 5
        sample_size = 5000

        class MixedModel(torch.nn.Module):
            def __init__(self) -> None:
                super(MixedModel, self).__init__()
                self.conv = torch.nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2)
                self.linear = torch.nn.Linear(dim_in, dim_out, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.view(x.shape[0], 1, -1)
                x = self.conv(x).view(x.shape[0], -1)
                return self.linear(x)

        @manual_cache_forward
        def custom_forward_method(self, hidden_states):
            if not hasattr(self, EKFAC_CACHE_KEY):
                # Normal forward pass
                hidden_states = hidden_states.view(hidden_states.shape[0], -1)
                return self.conv(self.linear(hidden_states))
            # Forward pass with caching i/o variables
            cache = getattr(self, EKFAC_CACHE_KEY)
            hidden_states = hidden_states.view(hidden_states.shape[0], 1, -1)
            x1 = self.conv(hidden_states).view(hidden_states.shape[0], -1)
            y1 = self.linear(x1)
            cache.input_hidden_pairs.append((x1, y1))
            return y1

        model = MixedModel()
        model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}

        x = torch.randn(sample_size, dim_in)
        y = torch.randn(sample_size, dim_out)
        v = torch.randn(5, dim_out, dim_in)

        @flatten_func(model, param_num=0)
        def f_ekfac(params, x, y):
            # The custom function should zero out gradient
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

            logits = torch.func.functional_call(model, params, x)
            return torch.mean((logits - y) ** 2, dim=-1)

        model.forward = types.MethodType(custom_forward_method, model)
        cache = MLPCache()
        setattr(model, EKFAC_CACHE_KEY, cache)

        ifvp_at_x_ekfac_func = ifvp_at_x_ekfac(
            f_ekfac,
            *(flatten_params(model_params), x, y),
            in_dims=(None, 0, 0),
            mlp_cache=cache,
            batch_size=128,
            damping=0.1,
        )

        ifvp_at_x_ekfac_func([[v]])

        # TODO: the current test is only checking whether the implementation
        # is error-free. More tests could be added.

    def test_ifvp_datainf(self):
        """Testing datainf functionality."""

        def loss_func(weight, inputs, labels):
            weights_resized = weight.view(3, 20)
            pred = inputs @ weights_resized.T
            loss = torch.nn.CrossEntropyLoss()
            return loss(pred, labels)

        def ce_single(weights, inputs, label):
            concatenated_weights = torch.cat(weights, dim=0)
            weights_resized = concatenated_weights.view(3, 20)
            pred = inputs @ weights_resized.T
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(pred.unsqueeze(0), label.unsqueeze(0))

        def corr(tensor1, tensor2):
            mean1 = torch.mean(tensor1)
            mean2 = torch.mean(tensor2)
            covariance = torch.mean((tensor1 - mean1) * (tensor2 - mean2))
            variance1 = torch.mean((tensor1 - mean1) ** 2)
            variance2 = torch.mean((tensor2 - mean2) ** 2)
            return covariance / (torch.sqrt(variance1) * torch.sqrt(variance2))

        random_data = torch.randn(500, 20)
        weights_layer1 = torch.randn(30, requires_grad=True)
        weights_layer2 = torch.randn(30, requires_grad=True)
        weights_flattened = torch.concat((weights_layer1, weights_layer2), dim=0)
        labels = torch.randint(0, 3, (500,))
        v_layer1 = torch.randn(30)
        v_layer2 = torch.randn(30)
        v_flattened = torch.concat((v_layer1, v_layer2), dim=0)
        tol = 0.9
        gt = ihvp_at_x_explicit(
            loss_func,
            *(weights_flattened, random_data, labels),
            argnums=0,
            regularization=0.07,
        )
        ifvp_datainf_func = ifvp_datainf(
            ce_single,
            0,
            (None, 0, 0),
            [0.07, 0.07],
        )
        ifvp = ifvp_datainf_func(
            ((weights_layer1, weights_layer2), random_data, labels),
            (v_layer1, v_layer2),
        )
        ifvp_datainf_at_x_func = ifvp_at_x_datainf(
            ce_single,
            0,
            (None, 0, 0),
            [0.07, 0.07],
            (weights_layer1, weights_layer2),
            random_data,
            labels,
        )
        assert corr(gt(v_flattened)[:30], ifvp[0]) > tol
        assert corr(gt(v_flattened)[30:], ifvp[1]) > tol
        assert (
            corr(gt(v_flattened)[:30], ifvp_datainf_at_x_func((v_layer1, v_layer2))[0])
            > tol
        )
        assert (
            corr(gt(v_flattened)[30:], ifvp_datainf_at_x_func((v_layer1, v_layer2))[1])
            > tol
        )

    def test_ifvp_datainf_nn(self):
        """Testing Datainf Functionality for a nn.Module."""
        model = torch.nn.Sequential(
            torch.nn.Linear(20, 6, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 3, bias=True),
        )
        model.eval()
        inputs = torch.randn((500, 20))
        labels = torch.randint(0, 3, (500,))
        model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
        v = (torch.randn(126), torch.randn(21))
        v_all = torch.cat(v, dim=0)

        @flatten_func(model, param_num=0)
        def f(params):
            yhat = torch.func.functional_call(model, params, inputs)
            loss = torch.nn.CrossEntropyLoss()
            return loss(yhat, labels)

        def ce(params, inputs, labels):
            param_dict = {}
            keys = list(model_params.keys())
            for i in range(len(keys)):
                param_dict[keys[i]] = params[i].view(model_params[keys[i]].shape)
            yhat = torch.func.functional_call(model, param_dict, inputs)
            loss = torch.nn.CrossEntropyLoss()
            return loss(yhat, labels)

        ifvp_explicit_at_x_func = ihvp_at_x_explicit(
            f,
            flatten_params(model_params),
            argnums=0,
            regularization=0.15,
        )
        ifvp_datainf_func = ifvp_datainf(
            ce,
            0,
            (None, 0, 0),
            [0.15, 0.15, 0.15, 0.15],
            param_layer_map=[0, 0, 1, 1],
        )

        def corr(tensor1, tensor2):
            mean1 = torch.mean(tensor1)
            mean2 = torch.mean(tensor2)
            covariance = torch.mean((tensor1 - mean1) * (tensor2 - mean2))
            variance1 = torch.mean((tensor1 - mean1) ** 2)
            variance2 = torch.mean((tensor2 - mean2) ** 2)
            return covariance / (torch.sqrt(variance1) * torch.sqrt(variance2))

        params = tuple([param.flatten() for param in model_params.values()])
        ifvp = ifvp_datainf_func((params, inputs, labels), v)
        tol = 0.8
        assert corr(ifvp[0], ifvp_explicit_at_x_func(v_all)[:126]) > tol
        # For layer 1 weights & biases
