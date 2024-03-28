#!/usr/bin/env python

'''
unit test for ihvp calculator
'''


from dattri.func.ihvp import ExactIHVP
import torch
import torch.nn as nn
from torch.func import functional_call


class Test_IVP:

    def test_ihvp_easy_cache(self):
        # a easy target which we can easily
        # calculate the hessian in closed form
        def target(x):
            return torch.sin(x).sum()
        ihvp = ExactIHVP(func=target, arg_num=0)

        x = torch.randn(2)
        vec = torch.randn(2)
        ihvp.cache(x)
        ihvp_result = ihvp.product(vec)

        assert torch.allclose(ihvp_result, torch.diag(-1 / x.sin()) @ vec)
