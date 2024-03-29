#!/usr/bin/env python

"""
ihvp (inverse hessian-vector product) calculation for an arbitrary function.

This module contains:
- `ihvp_direct`: Direct algorithm for ihvp.
"""

from typing import Callable

import torch
from torch import Tensor
from torch.func import hessian


def ihvp_direct(func: Callable, *args, argnums: int = 0):
    """
    Direct ihvp algorithm function.

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    Direct algorithm calcualte the hessian matrix explicitly and then use
    `torch.linagl.solve` for each vector production.

    :param func: A Python function that takes one or more arguments.
           Must return a single-element Tensor. The hessian will
           be calculated on this function.
    :param argnums: An integer default to 0. Specifies arguments to compute
           gradients with respect to.
    """
    hessian_tensor = hessian(func, argnums=argnums)(*args)

    def _ihvp_direct_func(vec: Tensor):
        return torch.linalg.solve(hessian_tensor, vec.T).T

    return _ihvp_direct_func
