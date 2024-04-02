"""Unit test for ihvp calculator."""

import torch

from dattri.func.ihvp import hvp, hvp_at_x, ihvp_at_x_cg, ihvp_at_x_explicit, ihvp_cg


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

        assert torch.allclose(ihvp(vec), (torch.diag(-1 / (2+y).sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)

    def test_hvp_at_x(self):
        """Test hvp_at_x."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(hvp_at_x(target, (x,), argnums=0, mode="rev-rev")(vec),
                             (torch.diag(-x.sin()) @ vec.T).T)
        assert torch.allclose(hvp_at_x(target, (x,), argnums=0, mode="rev-fwd")(vec),
                             (torch.diag(-x.sin()) @ vec.T).T)

    def test_hvp_at_x_argnums(self):
        """Test argnums of hvp_at_x."""

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([2])
        y = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(hvp_at_x(target, (x, y), argnums=1, mode="rev-rev")(vec),
                              (torch.diag(-(2+y).sin()) @ vec.T).T)
        assert torch.allclose(hvp_at_x(target, (x, y), argnums=1, mode="rev-rev")(vec),
                              (torch.diag(-(2+y).sin()) @ vec.T).T)

    def test_hvp_at_x_vmap(self):
        """Test vmap usage on hvp_at_x."""

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([2])
        y = torch.randn(2)
        vec = torch.randn(2)

        from torch.func import vmap

        hvp_at_x_func = hvp_at_x(target, (x, y), argnums=1, mode="rev-rev")

        assert torch.allclose(vmap(hvp_at_x_func)(torch.stack([vec for _ in range(5)])),
                              torch.stack([
                                  (torch.diag(-(2+y).sin()) @ vec.T).T
                                    for _ in range(5)]))

    def test_hvp(self):
        """Test hvp."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(hvp(target, argnums=0, mode="rev-rev")((x,), vec),
                             (torch.diag(-x.sin()) @ vec.T).T)
        assert torch.allclose(hvp(target, argnums=0, mode="rev-fwd")((x,), vec),
                             (torch.diag(-x.sin()) @ vec.T).T)

    def test_hvp_vmap(self):
        """Test vmap's usage on hvp."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(5, 2)
        vec = torch.randn(2)

        from torch.func import vmap

        def vmap_on_x(x):
            return hvp(target, argnums=0, mode="rev-rev")((x,), vec)

        torch.allclose(vmap(vmap_on_x)(x),
                       torch.stack([vmap_on_x(x[i]) for i in range(5)]))


    def test_ihvp_cg(self):
        """Test ihvp_cg/ihvp_cg_at_x."""

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_at_x_cg(target, x, argnums=0)

        assert torch.allclose(ihvp(vec),
                              (torch.diag(-1 / x.sin()) @ vec.T).T)
        assert torch.allclose(ihvp_cg(target, argnums=0)((x,), vec),
                              (torch.diag(-1 / x.sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)

    def test_ihvp_cg_argnum(self):
        """Test argnums of ihvp_cg/ihvp_cg_at_x."""

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([1])
        y = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_at_x_cg(target, x, y, argnums=1)

        assert torch.allclose(ihvp(vec),
                              (torch.diag(-1 / (1+y).sin()) @ vec.T).T)
        assert torch.allclose(ihvp_cg(target, argnums=1)((x, y), vec),
                              (torch.diag(-1 / (1+y).sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)
