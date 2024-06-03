"""Unit test for ihvp calculator."""

import types

import numpy as np
import torch
from torch.func import vmap

from dattri.func.ihvp import (
    EKFAC_CACHE_KEY,
    MLPCache,
    hvp,
    hvp_at_x,
    ihvp_arnoldi,
    ihvp_at_x_arnoldi,
    ihvp_at_x_cg,
    ihvp_at_x_datainf,
    ihvp_at_x_ekfac,
    ihvp_at_x_explicit,
    ihvp_at_x_lissa,
    ihvp_cg,
    ihvp_datainf,
    ihvp_lissa,
    manual_cache_forward,
)
from dattri.func.utils import flatten_func, flatten_params


class TestIHVP:
    """Test ihvp functions."""

    def test_ihvp_at_x_explicit(self):
        """Test ihvp_at_x_explicit."""

        def _target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_at_x_explicit(_target, x, argnums=0)

        assert torch.allclose(ihvp(vec), (torch.diag(-1 / x.sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)

    def test_ihvp_at_x_explicit_argnums(self):
        """Test argnums of ihvp_at_x_explicit."""

        def _target(x, y):
            return torch.sin(x + y).sum()

        x = 2
        y = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_at_x_explicit(_target, x, y, argnums=1)

        assert torch.allclose(ihvp(vec), (torch.diag(-1 / (2 + y).sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)

    def test_hvp_at_x(self):
        """Test hvp_at_x."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(
            hvp_at_x(target, (x,), argnums=0, mode="rev-rev")(vec),
            (torch.diag(-x.sin()) @ vec.T).T,
        )
        assert torch.allclose(
            hvp_at_x(target, (x,), argnums=0, mode="rev-fwd")(vec),
            (torch.diag(-x.sin()) @ vec.T).T,
        )

    def test_hvp_at_x_argnums(self):
        """Test argnums of hvp_at_x."""

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([2])
        y = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(
            hvp_at_x(target, (x, y), argnums=1, mode="rev-rev")(vec),
            (torch.diag(-(2 + y).sin()) @ vec.T).T,
        )
        assert torch.allclose(
            hvp_at_x(target, (x, y), argnums=1, mode="rev-rev")(vec),
            (torch.diag(-(2 + y).sin()) @ vec.T).T,
        )

    def test_hvp_at_x_vmap(self):
        """Test vmap usage on hvp_at_x."""

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([2])
        y = torch.randn(2)
        vec = torch.randn(2)

        hvp_at_x_func = hvp_at_x(target, (x, y), argnums=1, mode="rev-rev")

        assert torch.allclose(
            vmap(hvp_at_x_func)(torch.stack([vec for _ in range(5)])),
            torch.stack([(torch.diag(-(2 + y).sin()) @ vec.T).T for _ in range(5)]),
        )

    def test_hvp(self):
        """Test hvp."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(
            hvp(target, argnums=0, mode="rev-rev")((x,), vec),
            (torch.diag(-x.sin()) @ vec.T).T,
        )
        assert torch.allclose(
            hvp(target, argnums=0, mode="rev-fwd")((x,), vec),
            (torch.diag(-x.sin()) @ vec.T).T,
        )

    def test_hvp_vmap(self):
        """Test vmap's usage on hvp."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(5, 2)
        vec = torch.randn(2)

        def vmap_on_x(x):
            return hvp(target, argnums=0, mode="rev-rev")((x,), vec)

        torch.allclose(
            vmap(vmap_on_x)(x),
            torch.stack([vmap_on_x(x[i]) for i in range(5)]),
        )

    def test_ihvp_cg(self):
        """Test ihvp_cg/ihvp_cg_at_x."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_at_x_cg(target, x, argnums=0, max_iter=2)

        assert torch.allclose(
            ihvp(vec),
            (torch.diag(-1 / x.sin()) @ vec.T).T,
            rtol=1e-04,
            atol=1e-07,
        )
        assert torch.allclose(
            ihvp_cg(target, argnums=0, max_iter=2)((x,), vec),
            (torch.diag(-1 / x.sin()) @ vec.T).T,
            rtol=1e-04,
            atol=1e-07,
        )
        assert ihvp(vec).shape == (5, 2)

    def test_ihvp_cg_argnum(self):
        """Test argnums of ihvp_cg/ihvp_cg_at_x."""

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([1])
        y = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_at_x_cg(target, x, y, argnums=1, max_iter=2)

        assert torch.allclose(
            ihvp(vec),
            (torch.diag(-1 / (1 + y).sin()) @ vec.T).T,
            rtol=1e-04,
            atol=1e-07,
        )
        assert torch.allclose(
            ihvp_cg(target, argnums=1, max_iter=2)((x, y), vec),
            (torch.diag(-1 / (1 + y).sin()) @ vec.T).T,
            rtol=1e-04,
            atol=1e-07,
        )
        assert ihvp(vec).shape == (5, 2)

    def test_ihvp_arnoldi(self):
        """Test ihvp_arnoldi/ihvp_arnoldi_at_x."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_at_x_arnoldi(target, x, argnums=0)

        assert torch.allclose(
            ihvp(vec),
            (torch.diag(-1 / x.sin()) @ vec.T).T,
            rtol=1e-04,
            atol=1e-07,
        )
        assert torch.allclose(
            ihvp_arnoldi(target, argnums=0)((x,), vec),
            (torch.diag(-1 / x.sin()) @ vec.T).T,
            rtol=1e-04,
            atol=1e-07,
        )
        assert ihvp(vec).shape == (5, 2)

    def test_ihvp_cg_nn(self):
        """Test ihvp_at_x_cg and ihvp_cg for a nn forwarding function ."""
        # create a simple model with example data
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1),
        )
        data = (torch.randn(3), torch.randn(1))
        model.eval()

        @flatten_func(model, param_num=0)
        def f(params):
            yhat = torch.func.functional_call(model, params, data[0])
            return torch.mean((yhat - data[1]) ** 2)

        model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}

        v = torch.ones(16)
        ihvp_cg_func = ihvp_cg(f, argnums=0, regularization=1e-3, max_iter=10)
        ihvp_cg_at_x_func = ihvp_at_x_cg(
            f,
            flatten_params(model_params),
            argnums=0,
            regularization=1e-3,
            max_iter=10,
        )
        ihvp_explicit_at_x_func = ihvp_at_x_explicit(
            f,
            flatten_params(model_params),
            argnums=0,
            regularization=1e-3,
        )

        assert torch.allclose(
            ihvp_cg_at_x_func(v),
            ihvp_explicit_at_x_func(v),
            rtol=5e-02,
            atol=1e-05,
        )
        assert torch.allclose(
            ihvp_cg_func((flatten_params(model_params),), v),
            ihvp_explicit_at_x_func(v),
            rtol=5e-02,
            atol=1e-05,
        )

    def test_ihvp_lissa(self):
        """Test ihvp_lissa/ihvp_lissa_at_x."""

        def mse_loss(xs, ys, theta):
            return torch.mean((xs @ theta - ys) ** 2)

        # Create data
        data_size = (500, 2)
        theta = torch.randn(2)
        xs = torch.randn(data_size)
        noise = torch.normal(
            mean=torch.tensor(0),
            std=torch.tensor(0.05),
            size=(data_size[0],),
        )
        ys = xs @ theta + noise
        vec = torch.randn(1, 2)

        ihvp_lissa_func = ihvp_lissa(
            mse_loss,
            argnums=2,
            num_repeat=100,
            recursion_depth=100,
        )

        ihvp_lissa_at_x_func = ihvp_at_x_lissa(
            mse_loss,
            *(xs, ys, theta),
            in_dims=(0, 0, None),
            argnums=2,
            num_repeat=100,
            recursion_depth=100,
        )

        ihvp_explicit_at_x_func = ihvp_at_x_explicit(
            mse_loss,
            *(xs, ys, theta),
            argnums=2,
        )

        # Set a larger tolerance for LiSSA
        assert torch.allclose(
            ihvp_lissa_at_x_func(vec),
            ihvp_explicit_at_x_func(vec),
            atol=0.08,
        )
        assert torch.allclose(
            ihvp_lissa_func((xs, ys, theta), vec, in_dims=(0, 0, None)),
            ihvp_explicit_at_x_func(vec),
            atol=0.08,
        )

    def test_ihvp_lissa_batch_size(self):
        """Test ihvp_lissa/ihvp_lissa_at_x with different batch sizes."""

        def mse_loss(xs, ys, theta):
            return torch.mean((xs @ theta - ys) ** 2)

        # Create data
        data_size = (500, 2)
        theta = torch.randn(2)
        xs = torch.randn(data_size)
        noise = torch.normal(
            mean=torch.tensor(0),
            std=torch.tensor(0.05),
            size=(data_size[0],),
        )
        ys = xs @ theta + noise
        vec = torch.randn(1, 2)

        ihvp_lissa_func = ihvp_lissa(
            mse_loss,
            argnums=2,
            batch_size=4,
            num_repeat=10,
            recursion_depth=100,
        )

        ihvp_lissa_at_x_func = ihvp_at_x_lissa(
            mse_loss,
            *(xs, ys, theta),
            in_dims=(0, 0, None),
            argnums=2,
            batch_size=4,
            num_repeat=10,
            recursion_depth=100,
        )

        ihvp_explicit_at_x_func = ihvp_at_x_explicit(
            mse_loss,
            *(xs, ys, theta),
            argnums=2,
        )

        # Set a larger tolerance for LiSSA
        assert torch.allclose(
            ihvp_lissa_at_x_func(vec),
            ihvp_explicit_at_x_func(vec),
            atol=0.08,
        )
        assert torch.allclose(
            ihvp_lissa_func((xs, ys, theta), vec, in_dims=(0, 0, None)),
            ihvp_explicit_at_x_func(vec),
            atol=0.08,
        )

    def test_ihvp_ekfac(self):
        """Test ihvp_ekfac_at_x."""
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
            return torch.mean((logits - y)**2)

        @flatten_func(model, param_num=0)
        def f_ekfac(params, x, y):
            # The custom function should zero out gradient
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

            logits = torch.func.functional_call(model, params, x)
            return torch.mean((logits - y)**2, dim=-1)

        ihvp_explicit_at_x_func = ihvp_at_x_explicit(
            f,
            flatten_params(model_params),
            argnums=0,
        )

        ground_truth = ihvp_explicit_at_x_func(v.reshape(5, -1))

        model.forward = types.MethodType(custom_forward_method, model)
        cache = MLPCache()
        setattr(model, EKFAC_CACHE_KEY, cache)

        ihvp_at_x_ekfac_func = ihvp_at_x_ekfac(f_ekfac,
                                               *(flatten_params(model_params),
                                                 x, y),
                                               in_dims=(None, 0, 0),
                                               mlp_cache=cache,
                                               batch_size=128,
                                               damping=0.1)

        estimation = ihvp_at_x_ekfac_func([[v]])[0][0]
        for i in range(5):
            corr = np.corrcoef(ground_truth[i].detach().numpy(),
                               estimation[i].detach().numpy())[0, 1]
            assert corr > 0.95  # noqa: PLR2004


def test_ihvp_datainf():
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
        variance1 = torch.mean((tensor1 - mean1)**2)
        variance2 = torch.mean((tensor2 - mean2)**2)
        return covariance / (torch.sqrt(variance1)
                                * torch.sqrt(variance2))

    random_data = torch.randn(500, 20)
    weights_layer1 = torch.randn(30, requires_grad=True)
    weights_layer2 = torch.randn(30, requires_grad=True)
    weights_flattened = torch.concat((weights_layer1, weights_layer2), dim=0)
    labels = torch.randint(0, 3, (500,))
    v_layer1 = torch.randn(30)
    v_layer2 = torch.randn(30)
    v_flattened = torch.concat((v_layer1, v_layer2), dim=0)
    tol = 0.9
    gt = ihvp_at_x_explicit(loss_func,
                            *(weights_flattened, random_data, labels),
                            argnums=0,
                            regularization=0.07)
    ihvp_datainf_func = ihvp_datainf(
        ce_single,
        0,
        (None, 0, 0),
        [0.07, 0.07],
    )
    ihvp = ihvp_datainf_func(((weights_layer1, weights_layer2),
                            random_data, labels),
                            (v_layer1, v_layer2))
    ihvp_datainf_at_x_func = ihvp_at_x_datainf(ce_single,
                                    0,
                                    (None, 0, 0),
                                    [0.07, 0.07],
                                    (weights_layer1, weights_layer2),
                                    random_data,
                                    labels)
    assert (corr(gt(v_flattened)[:30], ihvp[0]) > tol)
    assert (corr(gt(v_flattened)[30:], ihvp[1]) > tol)
    assert (corr(gt(v_flattened)[:30],
                ihvp_datainf_at_x_func((v_layer1, v_layer2))[0]) > tol)
    assert (corr(gt(v_flattened)[30:],
                ihvp_datainf_at_x_func((v_layer1, v_layer2))[1]) > tol)


def test_ihvp_datainf_nn():
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
    ihvp_explicit_at_x_func = ihvp_at_x_explicit(
            f,
            flatten_params(model_params),
            argnums=0,
            regularization=0.15,
    )
    ihvp_datainf_func = ihvp_datainf(
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
        variance1 = torch.mean((tensor1 - mean1)**2)
        variance2 = torch.mean((tensor2 - mean2)**2)
        return covariance / (torch.sqrt(variance1)
                                * torch.sqrt(variance2))
    params = tuple([param.flatten() for param in model_params.values()])
    ihvp = ihvp_datainf_func((params, inputs, labels), v)
    tol = 0.9
    assert (corr(ihvp[0], ihvp_explicit_at_x_func(v_all)[:126]) > tol)
    # For layer 1 weights & biases
