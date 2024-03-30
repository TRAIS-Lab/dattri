"""IHVP (inverse hessian-vector product) calculation.

This module contains:
- `ihvp_explicit`: IHVP via explicit Hessian calculation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Tuple, Union

from functools import partial

import torch
from torch import Tensor
from torch.func import hessian, grad, vjp, jvp


def hvp(func: Callable,
        primals: Tuple[torch.Tensor, ...],
        tangents: torch.Tensor,
        argnums: int = 0,
        mode: str = "rev-only"):
    """
    Hessian Vector Product(HVP) calculation function.

    This function returns a function that could calculate the hessian with
    respect to the argument user specified.

    :param func: A Python function that takes one or more arguments.
           Must return a single-element Tensor. The hessian will
           be calculated on this function.
    :param primals: Positional arguments to func that must all be Tensors.
           The returned function will also be computing the derivative with
           respect to these arguments
    :param tangents: The “vector” for which Hessian Vector Product is computed.
    :param argnums: An integer default to 0. Specifies arguments to compute
           gradients with respect to.
    :param mode: A string default to `rev-only`. Specifies how to calculate
           hvp function. Either `rev-only` or `rev-fwd` should be put here.
    """
    if mode not in ["rev-only", "rev-fwd"]:
        raise IHVPUsageException("`mode` should be either"
                                 "'rev-only' or 'rev-fwd'.")

    if mode == "rev-only":
        # pylint: disable=unbalanced-tuple-unpacking
        _, vjp_fn = vjp(grad(func, argnums=argnums), *primals)
        return vjp_fn(tangents)[argnums]
    else:
        # patch the tangents, make zero for other argnums
        # e.g.,
        # if argnums = 1, and len(primals) = 3, then it should be patched to
        # [torch.zeros(*primals[0].shape),
        # tangents,
        # torch.zeros(*primals[2].shape)]
        if len(primals) > 1:
            tangents_patched = []
            for i in range(len(primals)):
                if i != argnums:
                    tangents_patched.append(torch.zeros(*primals[i].shape))
                else:
                    tangents_patched.append(tangents)
            tangents_patched = tuple(tangents_patched)
        else:
            tangents_patched = (tangents,)

        return jvp(grad(func, argnums=argnums), primals, tangents_patched)[1]


def ihvp_explicit(func: Callable,
                  *args,
                  argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
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


def ihvp_cg(func: Callable,
            *args,
            argnums: int = 0,
            max_iter: int = 100,
            tol: float = 1e-7,
            mode: str = "rev-only"):
    """
    Conjugate Gradient Descent ihvp algorithm function.

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    Conjugate Gradient Descent algorithm calcualte the hvp function and use
    it iteratively through Conjugate Gradient

    :param func: A Python function that takes one or more arguments.
           Must return a single-element Tensor. The hessian will
           be calculated on this function.
    :param argnums: An integer default to 0. Specifies arguments to compute
           gradients with respect to.
    :param max_iter: An integer default 100. Specifies the maximum iteration
           to calculate the ihvp.
    :param tol: A float default to 1e-7. Specifies the break condition that
           decide if the algorithm has converged.
    :param mode: A string default to `rev-only`. Specifies how to calculate
           hvp function. Either `rev-only` or `rev-fwd` should be put here.
    """
    hvp_helper_func = partial(hvp, func, (*args, ), argnums=argnums, mode=mode)

    def _ihvp_cg(vec: Tensor):
        # algorithm refer to
        # https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/conjugate_direction_methods.pdf

        if vec.ndim < 2:
            vec = vec.unsqueeze(0)
        batch_ihvp_cg = []

        for i in range(vec.shape[0]):
            x_pre = torch.clone(vec[i, :])
            x = x_pre
            g_pre = vec[i, :] - hvp_helper_func(x)
            d = d_pre = g_pre

            for _ in range(max_iter):
                if torch.norm(g_pre) < tol:
                    break
                ad = hvp_helper_func(d)
                alpha = torch.dot(g_pre, d_pre) / torch.dot(d, ad)
                x += alpha * d
                g = g_pre - alpha * ad

                beta = torch.dot(g, g) / torch.dot(g_pre, g_pre)

                g_pre = d_pre = g
                d = g + beta * d
            batch_ihvp_cg.append(x)

        return torch.stack(batch_ihvp_cg)

    return _ihvp_cg


class IHVPUsageException(Exception):
    """The usage exception class for ihvp module."""

    pass
