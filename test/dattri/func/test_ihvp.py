"""Unit test for ihvp calculator."""
import torch

from dattri.func.ihvp import ihvp_explicit


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
