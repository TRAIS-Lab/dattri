"""IHVP (inverse hessian-vector product) calculation.

This module contains:
- `ihvp_explicit`: IHVP via explicit Hessian calculation.
"""

from collections.abc import Callable
from typing import Union

import torch
from torch import Tensor
from torch.func import hessian


def ihvp_explicit(func: Callable,
                  *args,
                  argnums: Union[int, tuple[int, ...]] = 0) -> Callable:
    """IHVP via explicit Hessian calculation.

    IHVP stands for inverse-hessian-vector product. For a given function
    `func`, this method first calculates the Hessian matrix explicitly
    and then wraps the Hessian in a function that uses `torch.linalg.solve` to
    calculate the IHVP for any given vector.

    Args:
        func (Callable): A function taking one or more arguments and returning
            a single-element Tensor. The Hessian will be calculated based on
            this function.
        *args: List of arguments for `func`.
        argnums (int or Tuple[int], optional): An integer or a tuple of integers
            deciding which arguments in `*args` to get the Hessian with respect
            to. Default: 0.

    Returns:
        A function that takes a vector `vec` and returns the IHVP of the Hessian
        of `func` and `vec`.

    Note:
        This method stores the Hessian matrix explicitly and is not computationally
        efficient.
    """
    hessian_tensor = hessian(func, argnums=argnums)(*args)

    def _ihvp_direct_func(vec: Tensor) -> Tensor:
        """The IHVP function based on `hessian_tensor`.

        Args:
            vec (Tensor): A vector with the same dimension as the first dim of
                `hessian_tensor`.

        Returns:
            The IHVP value, i.e., inverse of `hessian_tensor` times `vec`.
        """
        return torch.linalg.solve(hessian_tensor, vec.T).T

    return _ihvp_direct_func
