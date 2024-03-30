"""Unit test for ihvp calculator."""
import torch

from dattri.func.ihvp import ihvp_explicit, hvp, ihvp_cg


class TestIHVP:
    """Test ihvp functions."""

    def test_ihvp_explicit(self):
        """Test ihvp_explicit."""

        def _target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_explicit(_target, x, argnums=0)

        assert torch.allclose(ihvp(vec), (torch.diag(-1 / x.sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)

    def test_ihvp_explicit_argnums(self):
        """Test argnums of ihvp_explicit."""

        def _target(x, y):
            return torch.sin(x + y).sum()

        x = 2
        y = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_explicit(_target, x, y, argnums=1)

        assert torch.allclose(ihvp(vec), (torch.diag(-1 / (2+y).sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)
    
    def test_hvp_rev_only(self):

        def target(x):
            return torch.sin(x).sum()
        
        x = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(hvp(target, (x,), vec, argnums=0, mode="rev-only"),
                             (torch.diag(-x.sin()) @ vec.T).T)

    def test_hvp_rev_fwd(self):

        def target(x):
            return torch.sin(x).sum()
        
        x = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(hvp(target, (x,), vec, argnums=0, mode="rev-fwd"),
                             (torch.diag(-x.sin()) @ vec.T).T)

    def test_hvp_rev_only_argnums(self):

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([2])
        y = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(hvp(target, (x, y), vec, argnums=1, mode="rev-only"),
                              (torch.diag(-(2+y).sin()) @ vec.T).T)

    def test_hvp_rev_only_argnums(self):

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([2])
        y = torch.randn(2)
        vec = torch.randn(2)

        assert torch.allclose(hvp(target, (x, y), vec, argnums=1, mode="rev-fwd"),
                              (torch.diag(-(2+y).sin()) @ vec.T).T)
        
    def test_hvp_vmap(self):

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([2])
        y = torch.randn(2)
        vec = torch.randn(2)

        from functools import partial
        from torch.func import vmap

        vmap_hvp = partial(hvp, target, (x, y), argnums=1, mode="rev-only")

        assert torch.allclose(vmap(vmap_hvp)(torch.stack([vec for _ in range(5)])),
                              torch.stack([(torch.diag(-(2+y).sin()) @ vec.T).T for _ in range(5)]))
    
    def test_ihvp_cg(self):
        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_cg(target, x, argnums=0)

        assert torch.allclose(ihvp(vec), (torch.diag(-1 / x.sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)

    def test_ihvp_cg_argnum(self):

        def target(x, y):
            return torch.sin(x + y).sum()

        x = torch.Tensor([1])
        y = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_cg(target, x, y, argnums=1)

        assert torch.allclose(ihvp(vec), (torch.diag(-1 / (1+y).sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)
