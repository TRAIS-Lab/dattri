#!/usr/bin/env python

'''
unit test for ihvp calculator
'''


from dattri.func.ihvp import ihvp_direct
import torch


class Test_IVP:

    def test_ihvp_easy_cache(self):

        def target(x):
            return torch.sin(x).sum()

        x = torch.randn(2)
        vec = torch.randn(5, 2)
        ihvp = ihvp_direct(target, x, argnums=0)

        assert torch.allclose(ihvp(vec), (torch.diag(-1 / x.sin()) @ vec.T).T)
        assert ihvp(vec).shape == (5, 2)
