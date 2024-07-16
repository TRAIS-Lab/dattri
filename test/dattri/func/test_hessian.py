"""Unit test for ihvp calculator."""

import torch
from torch.func import vmap

from dattri.func.hessian import (
    hvp,
    hvp_at_x,
    ihvp_arnoldi,
    ihvp_at_x_arnoldi,
    ihvp_at_x_cg,
    ihvp_at_x_explicit,
    ihvp_at_x_lissa,
    ihvp_cg,
    ihvp_lissa,
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
