"""Unit test for ihvp calculator."""

import torch
from torch.func import vmap
import sys
sys.path.append("./dattri")
from dattri.func.ihvp import (
    hvp,
    hvp_at_x,
    ihvp_arnoldi,
    ihvp_at_x_arnoldi,
    ihvp_at_x_cg,
    ihvp_at_x_explicit,
    ihvp_at_x_lissa,
    ihvp_at_x_datainf,
    test_ihvp_gt,
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
    
        def test_ihvp_datainf(self):
        def loss_func(weight, inputs, labels, reg):
            weights_resized = weight.view(3, 20)
            pred = inputs @ weights_resized.T
            loss = torch.nn.CrossEntropyLoss()
            return loss(pred, labels) + reg * torch.sum(weight**2)

        def _compute_damping(avg_grad_dict, train_grad_dict, lambda_const_param=10):
            regularization = []
            for weight_name in avg_grad_dict:
                S = torch.zeros(len(train_grad_dict))
                for tr_id in range(len(train_grad_dict)):
                    tmp_grad = train_grad_dict[tr_id][weight_name]
                    S[tr_id] = torch.mean(tmp_grad**2)
                lambda_const = torch.mean(S) / lambda_const_param
                regularization.append(lambda_const)
            return regularization

        def corr(tensor1, tensor2):
            mean1 = torch.mean(tensor1)
            mean2 = torch.mean(tensor2)
            covariance = torch.mean((tensor1 - mean1) * (tensor2 - mean2))
            variance1 = torch.mean((tensor1 - mean1) ** 2)
            variance2 = torch.mean((tensor2 - mean2) ** 2)
            correlation = covariance / (torch.sqrt(variance1) * torch.sqrt(variance2))
            return correlation

        def get_test_grad(random_data, weights, labels):
            size = random_data.shape[0]
            grads = []
            for i in range(size):
                if weights.grad is not None:
                    weights.grad.zero_()  # Zero out the previous gradients
                loss = loss_func(weights, random_data[i], labels[i], reg=0)
                loss.backward()
                grad_dict = dict()
                grad_dict["layer1"] = weights.grad.flatten()
                grads.append(grad_dict)
            return grads

        random_data = torch.randn(500, 20)
        weights = torch.randn(3 * 20, requires_grad=True)
        labels = torch.randint(0, 3, (500,))
        v = torch.randn(
            60,
        )

        vect = dict()
        vect["layer1"] = v
        reg = _compute_damping(vect, get_test_grad(random_data, weights, labels))
        gt = test_ihvp_gt(random_data, weights, labels, v.T, reg[0])
        ihvp_func = ihvp_at_x_datainf(
            get_test_grad, 1, reg, random_data, weights, labels
        )
        datainf = ihvp_func(vect)
        assert corr(gt, datainf["layer1"]) > 0.9