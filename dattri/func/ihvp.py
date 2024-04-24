"""IHVP (inverse hessian-vector product) calculation.

This module contains:
- `hvp`: Calculate the Hessian Vector Product (HVP) of a function.
- `hvp_at_x`: Calculate the Hessian Vector Product (HVP) of a function with fixed x.
- `ihvp_at_x_explicit`: IHVP via explicit Hessian calculation.
- `ihvp_cg`: Conjugate Gradient Descent ihvp algorithm function.
- `ihvp_at_x_cg`: Conjugate Gradient Descent ihvp algorithm function with fixed x.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import List, Tuple, Union

import warnings

import torch
from torch import Tensor
from torch.func import grad, hessian, jvp, vjp


def hvp(func: Callable,
        argnums: int = 0,
        mode: str = "rev-rev",
        regularization: float = 0.0) -> Callable:
    """Hessian Vector Product(HVP) calculation function.

    This function takes the func where hessian is carried out and return a
    function takes x (the argument of func) and a vector v to calculate the
    hessian-vector production.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The hessian will
            be calculated on this function. The positional arguments to func
            must all be Tensors.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute hessian with respect to.
        mode (str): The auto diff mode, which can have one of the following values:
            - rev-rev: calculate the hessian with two reverse-mode auto-diff. It has
                       better compatibility while cost more memory.
            - rev-fwd: calculate the hessian with the composing of reverse-mode and
                       forward-mode. It's more memory-efficient but may not be supported
                       by some operator.
        regularization (float): A float default to 0.0. Specifies the regularization
            term to be added to the Hessian vector product, which is useful for the
            later inverse calculation if the Hessian matrix is singular or
            ill-conditioned. Specifically, the regularization term is
            `regularization * v`.

    Returns:
        A function that takes a tuple of Tensor `x` as the arguments of func and
        a vector `v` and returns the HVP of the Hessian of `func` and `v`.

    Note:
        This method does not fix the x. It's suitable if you have multiple `x` for
        the hvp calculation. If you have a fixed x please consider using `hvp_at_x`.

    Raises:
        IHVPUsageError: If mode is not one of "rev-rev" and "rev-fwd".
    """
    if mode not in ["rev-rev", "rev-fwd"]:
        error_msg = "`mode` should be either 'rev-rev' or 'rev-fwd'."
        raise IHVPUsageError(error_msg)

    grad_func = grad(func, argnums=argnums)
    if mode == "rev-rev":
        def _hvp_func(x: Tuple[torch.Tensor, ...], v: Tensor) -> Tensor:
            """The HVP function based on func.

            Args:
                x (Tuple[torch.Tensor, ...]): The function will computed the
                    hessian matrix with respect to these arguments.
                v (Tensor): A vector with the same dimension as the first dim of
                    `hessian_tensor`.

            Returns:
                (Tensor) The hessian vector production.
            """
            _, vjp_fn = vjp(grad_func, *x)
            return vjp_fn(v)[argnums] + regularization * v
    else:
        def _hvp_func(x: Tuple[torch.Tensor, ...], v: Tensor) -> Tensor:
            """The HVP function based on func.

            Args:
                x (Tuple[torch.Tensor, ...]): The function will computed the
                    hessian matrix with respect to these arguments.
                v (Tensor): A vector with the same dimension as the first dim of
                    `hessian_tensor`.

            Returns:
                (Tensor) The hessian vector production.
            """
            if len(x) > 1:
                v_patched = []
                for i in range(len(x)):
                    if i != argnums:
                        v_patched.append(torch.zeros(*x[i].shape))
                    else:
                        v_patched.append(v)
                v_patched = tuple(v_patched)
            else:
                v_patched = (v,)

            return jvp(grad_func, x, v_patched)[1] + regularization * v

    return _hvp_func


def hvp_at_x(func: Callable,
             x: Tuple[torch.Tensor, ...],
             argnums: int = 0,
             mode: str = "rev-rev",
             regularization: float = 0.0) -> Callable:
    """Hessian Vector Product(HVP) calculation function (with fixed x).

    This function returns a function that takes a vector `v` and calculate
    the hessian-vector production.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The hessian will
            be calculated on this function. The positional arguments to func
            must all be Tensors.
        x (Tuple[torch.Tensor, ...]): The returned function will computed the
            hessian matrix with respect to these arguments. `argnums` indicate
            which of the input `x` is used as primal.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute hessian with respect to.
        mode (str): The auto diff mode, which can have one of the following values:
            - rev-rev: calculate the hessian with two reverse-mode auto-diff. It has
                       better compatibility while cost more memory.
            - rev-fwd: calculate the hessian with the composing of reverse-mode and
                       forward-mode. It's more memory-efficient but may not be supported
                       by some operator.
        regularization (float): A float default to 0.0. Specifies the regularization
            term to be added to the Hessian vector product, which is useful for the
            later inverse calculation if the Hessian matrix is singular or
            ill-conditioned. Specifically, the regularization term is
            `regularization * v`.

    Returns:
        A function that takes a vector `v` and returns the HVP of the Hessian
        of `func` and `v`.

    Note:
        This method does fix the x to avoid some additional computation. If you have
        multiple x and want to use vmap to accelerate the computation, please consider
        using `hvp`.

    Raises:
        IHVPUsageError: If mode is not one of "rev-rev" and "rev-fwd".
    """
    # support tuple of int for argnums

    if mode not in ["rev-rev", "rev-fwd"]:
        error_msg = "`mode` should be either 'rev-rev' or 'rev-fwd'."
        raise IHVPUsageError(error_msg)

    grad_func = grad(func, argnums=argnums)
    if mode == "rev-rev":
        # pylint: disable=unbalanced-tuple-unpacking
        _, vjp_fn = vjp(grad_func, *x)

        def _hvp_at_x_func(v: Tensor) -> Tensor:
            """The HVP function based on func.

            Args:
                v (Tensor): A vector with the same dimension as the first dim of
                    `hessian_tensor`.

            Returns:
                (Tensor) The hessian vector production.
            """
            return vjp_fn(v)[argnums] + regularization * v
    else:
        # patch the v, make zero for other input
        # e.g.,
        # if argnums = 1, and len(x) = 3, then it should be patched to
        # [torch.zeros(*x[0].shape),
        # v,
        # torch.zeros(*x[2].shape)]
        def _hvp_at_x_func(v: Tensor) -> Tensor:
            """The HVP function based on func.

            Args:
                v (Tensor): A vector with the same dimension as the first dim of
                    `hessian_tensor`.

            Returns:
                (Tensor) The hessian vector production.
            """
            if len(x) > 1:
                v_patched = []
                for i in range(len(x)):
                    if i != argnums:
                        v_patched.append(torch.zeros(*x[i].shape))
                    else:
                        v_patched.append(v)
                v_patched = tuple(v_patched)
            else:
                v_patched = (v,)

            return jvp(grad_func, x, v_patched)[1] + regularization * v

    return _hvp_at_x_func


def ihvp_at_x_explicit(func: Callable,
                       *x,
                       argnums: Union[int, Tuple[int, ...]] = 0,
                       regularization: float = 0.0) -> Callable:
    """IHVP via explicit Hessian calculation.

    IHVP stands for inverse-hessian-vector product. For a given function
    `func`, this method first calculates the Hessian matrix explicitly
    and then wraps the Hessian in a function that uses `torch.linalg.solve` to
    calculate the IHVP for any given vector.

    Args:
        func (Callable): A function taking one or more arguments and returning
            a single-element Tensor. The Hessian will be calculated based on
            this function.
        *x: List of arguments for `func`.
        argnums (int or Tuple[int], optional): An integer or a tuple of integers
            deciding which arguments in `*x` to get the Hessian with respect
            to. Default: 0.
        regularization (float): A float default to 0.0. Specifies the
            regularization term to be added to the Hessian matrix. This is useful
            when the Hessian matrix is singular or ill-conditioned. The regularization
            term is `regularization * I`, where `I` is the identity matrix directly
            added to the Hessian matrix.

    Returns:
        A function that takes a vector `v` and returns the IHVP of the Hessian
        of `func` and `v`.

    Note:
        This method stores the Hessian matrix explicitly and is not computationally
        efficient.
    """
    hessian_tensor = hessian(func, argnums=argnums)(*x)

    def _ihvp_at_x_explicit_func(v: Tensor) -> Tensor:
        """The IHVP function based on `hessian_tensor`.

        Args:
            v (Tensor): A vector with the same dimension as the first dim of
                `hessian_tensor`.

        Returns:
            The IHVP value, i.e., inverse of `hessian_tensor` times `vec`.
        """
        return torch.linalg.solve(hessian_tensor +
                torch.eye(hessian_tensor.shape[0]) * regularization, v.T).T

    return _ihvp_at_x_explicit_func


def ihvp_cg(func: Callable,
            argnums: int = 0,
            max_iter: int = 100,
            tol: float = 1e-7,
            mode: str = "rev-rev",
            regularization: float = 0.0) -> Callable:
    """Conjugate Gradient Descent ihvp algorithm function.

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    Conjugate Gradient Descent algorithm calculate the hvp function and use
    it iteratively through Conjugate Gradient.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The hessian will
            be calculated on this function.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse hessian with respect to.
        max_iter (int): An integer default 100. Specifies the maximum iteration
            to calculate the ihvp through Conjugate Gradient Descent.
        tol (float): A float default to 1e-7. Specifies the break condition that
            decide if the algorithm has converged. If the torch.norm of residual
            is less than tol, then the algorithm is truncated.
        mode (str): The auto diff mode, which can have one of the following values:
            - rev-rev: calculate the hessian with two reverse-mode auto-diff. It has
                       better compatibility while cost more memory.
            - rev-fwd: calculate the hessian with the composing of reverse-mode and
                       forward-mode. It's more memory-efficient but may not be supported
                       by some operator.
        regularization (float): A float default to 0.0. Specifies the regularization
            term to be added to the Hessian vector product, which is useful for the
            later inverse calculation if the Hessian matrix is singular or
            ill-conditioned. Specifically, the regularization term is
            `regularization * v`.

    Returns:
        A function that takes a tuple of Tensor `x` and a vector `v` and returns
        the IHVP of the Hessian of `func` and `v`.
    """

    def _ihvp_cg_func(x: Tuple[torch.Tensor, ...], v: Tensor) -> Tensor:
        """The IHVP function using CG.

        Args:
            x (Tuple[torch.Tensor, ...]): The function will computed the
                inverse hessian matrix with respect to these arguments.
            v (Tensor): The vector to be produced on the inverse hessian matrix.

        Returns:
            The IHVP value.
        """
        return ihvp_at_x_cg(func, *x, argnums=argnums,
                            max_iter=max_iter, tol=tol,
                            mode=mode, regularization=regularization)(v)

    return _ihvp_cg_func


def ihvp_at_x_cg(func: Callable,
                 *x,
                 argnums: int = 0,
                 max_iter: int = 100,
                 tol: float = 1e-7,
                 mode: str = "rev-rev",
                 regularization: float = 0.0) -> Callable:
    """Conjugate Gradient Descent ihvp algorithm function (with fixed x).

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    Conjugate Gradient Descent algorithm calculated the hvp function and use
    it iteratively through Conjugate Gradient.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The hessian will
            be calculated on this function.
        *x: List of arguments for `func`.
        argnums (int): An integer default to 0. SSpecifies which argument of func
            to compute inverse hessian with respect to.
        max_iter (int): An integer default 100. Specifies the maximum iteration
            to calculate the ihvp through Conjugate Gradient Descent.
        tol (float): A float default to 1e-7. Specifies the break condition that
            decide if the algorithm has converged. If the torch.norm of residual
            is less than tol, then the algorithm is truncated.
        mode (str): The auto diff mode, which can have one of the following values:
            - rev-rev: calculate the hessian with two reverse-mode auto-diff. It has
                       better compatibility while cost more memory.
            - rev-fwd: calculate the hessian with the composing of reverse-mode and
                       forward-mode. It's more memory-efficient but may not be supported
                       by some operator.
        regularization (float): A float default to 0.0. Specifies the regularization
            term to be added to the Hessian vector product, which is useful for the
            later inverse calculation if the Hessian matrix is singular or
            ill-conditioned. Specifically, the regularization term is
            `regularization * v`.

    Returns:
        A function that takes a vector `v` and returns the IHVP of the Hessian
        of `func` and `v`.
    """
    hvp_at_x_func = hvp_at_x(func, x=(*x, ), argnums=argnums,
                             mode=mode, regularization=regularization)

    def _ihvp_cg_func(v: Tensor) -> Tensor:
        """The IHVP function using CG.

        Args:
            v (Tensor): The vector to be produced on the inverse hessian matrix.

        Returns:
            The IHVP value.
        """
        # algorithm refer to
        # https://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/conjugate_direction_methods.pdf

        if v.ndim == 1:
            v = v.unsqueeze(0)
        batch_ihvp_cg = []

        for i in range(v.shape[0]):
            x_pre = torch.clone(v[i, :])
            x = x_pre
            g_pre = v[i, :] - hvp_at_x_func(x)
            d = d_pre = g_pre

            for _ in range(max_iter):
                if torch.norm(g_pre) < tol:
                    break
                ad = hvp_at_x_func(d)
                alpha = torch.dot(g_pre, d_pre) / torch.dot(d, ad)
                x += alpha * d
                g = g_pre - alpha * ad

                beta = torch.dot(g, g) / torch.dot(g_pre, g_pre)

                g_pre = d_pre = g
                d = g + beta * d
            batch_ihvp_cg.append(x)

        return torch.stack(batch_ihvp_cg)

    return _ihvp_cg_func


def ihvp_lissa(func: Callable,
               argnums: int = 0,
               num_repeat: int = 10,
               recursion_depth: int = 5000,
               mode: str = "rev-rev") -> Callable:
    """LiSSA ihvp algorithm function.

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    LiSSA algorithm approximates the ihvp function by averaging multiple samples.
    The samples are estimated by recursion based on Taylor expansion.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The hessian will
            be estimated on this function.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse hessian with respect to.
        num_repeat (int): An integer default 10. Specifies the number of samples
            of the hvp approximation to average on.
        recursion_depth (int): A integer default to 5000. Specifies the number of
            recursions used to estimate each ihvp sample.
        mode (str): The auto diff mode, which can have one of the following values:
            - rev-rev: calculate the hessian with two reverse-mode auto-diff. It has
                       better compatibility while cost more memory.
            - rev-fwd: calculate the hessian with the composing of reverse-mode and
                       forward-mode. It's more memory-efficient but may not be supported
                       by some operator.

    Returns:
        A function that takes a list of tuples of Tensor `x` and a vector `v` and
        returns the IHVP of the Hessian of `func` and `v`.
    """

    def _ihvp_lissa_func(x: Tuple[torch.Tensor, ...],
                      v: Tensor,
                      in_dims: Tuple = None) -> Tensor:
        """The IHVP function using CG.

        Args:
            x (Tuple[torch.Tensor, ...]): The function will computed the
                inverse hessian matrix with respect to these arguments.
            v (Tensor): The vector to be produced on the inverse hessian matrix.
            in_dims (Tuple): A tuple with the same shape as *x. Indicating which 
                dimension should be considered as batch size dimension.

        Returns:
            The IHVP value.
        """
        return ihvp_at_x_lissa(func,
                               *x,
                               in_dims=in_dims,
                               argnums=argnums,
                               num_repeat=num_repeat,
                               recursion_depth=recursion_depth,
                               mode=mode)(v)

    return _ihvp_lissa_func


def ihvp_at_x_lissa(func: Callable,
                    *x,
                    in_dims: Tuple = None,
                    argnums: int = 0,
                    num_repeat: int = 10,
                    recursion_depth: int = 5000,
                    mode: str = "rev-rev") -> Callable:
    """LiSSA ihvp algorithm function (with fixed x).

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    LiSSA algorithm approximates the ihvp function by averaging multiple samples.
    The samples are estimated by recursion based on Taylor expansion.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The hessian will
            be estimated on this function.
        *x: List of arguments for `func`.
        in_dims (Tuple): A tuple with the same shape as *x, indicating which 
            dimension should be considered as batch size dimension. Take the
            first dimension as batch size dimension by default.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse hessian with respect to.
        num_repeat (int): An integer default 10. Specifies the number of samples
            of the hvp approximation to average on.
        recursion_depth (int): A integer default to 5000. Specifies the number of
            recursions used to estimate each ihvp sample.
        mode (str): The auto diff mode, which can have one of the following values:
            - rev-rev: calculate the hessian with two reverse-mode auto-diff. It has
                       better compatibility while cost more memory.
            - rev-fwd: calculate the hessian with the composing of reverse-mode and
                       forward-mode. It's more memory-efficient but may not be supported
                       by some operator.

    Returns:
        A function that takes a vector `v` and returns the IHVP of the Hessian
        of `func` and `v`.
    """
    if in_dims is None:
        in_dims = (0,) * len(x)

    # Check batch size mismatch
    batch_size = None
    for x_in, dim in zip(x, in_dims):
        if dim is not None:
            if batch_size is None:
                batch_size = x_in.shape[dim]
            else:
                assert batch_size == x_in.shape[dim], "Batch size mismatch!"

    assert batch_size is not None, "All inputs are identical!"

    # Create input list
    input_list = []
    for i in range(batch_size):
        new_input = [
            x_in.select(dim, i) if dim is not None else x_in
            for x_in, dim in zip(x, in_dims)
        ]
        input_list.append(tuple(new_input))

    if recursion_depth > len(input_list):
        warning_message = 'The recursion depth is greater than number of samples. " \
            "Please consider using other methods!'
        warnings.warn(warning_message, Warning, stacklevel=2)
        recursion_depth = len(input_list)

    def _ihvp_lissa_func(v: Tensor) -> Tensor:
        """The IHVP function using LiSSA.

        Args:
            v (Tensor): The vector to be produced on the inverse hessian matrix.

        Returns:
            The IHVP value.
        """
        if v.ndim == 1:
            v = v.unsqueeze(0)
        batch_ihvp_lissa = []

        for i in range(v.shape[0]):
            damping, scaling = 0.0, 50.0
            ihvp_estimations = []
            for _ in range(num_repeat):
                sampled_indices = torch.randperm(len(input_list))[:recursion_depth]
                hvp_func_list = [
                    hvp_at_x(func, x=input_list[idx], argnums=argnums, mode=mode)
                    for idx in sampled_indices
                ]

                curr_estimate = v[i, :].detach().clone()  # No gradient on v
                for hvp_func in hvp_func_list:
                    hvp = hvp_func(curr_estimate)
                    curr_estimate = (v[i, :]
                                     + (1 - damping) * curr_estimate
                                     - hvp / scaling)

                ihvp_estimations.append(curr_estimate / scaling)

            ihvp_value = torch.mean(torch.stack(ihvp_estimations), dim=0)
            batch_ihvp_lissa.append(ihvp_value)

        return torch.stack(batch_ihvp_lissa)

    return _ihvp_lissa_func


class IHVPUsageError(Exception):
    """The usage exception class for ihvp module."""
