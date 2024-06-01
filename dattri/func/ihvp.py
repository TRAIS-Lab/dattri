"""IHVP (inverse hessian-vector product) calculation.

This module contains:
- `hvp`: Calculate the Hessian Vector Product (HVP) of a function.
- `hvp_at_x`: Calculate the Hessian Vector Product (HVP) of a function with fixed x.
- `ihvp_at_x_explicit`: IHVP via explicit Hessian calculation.
- `ihvp_cg`: Conjugate Gradient Descent ihvp algorithm function.
- `ihvp_at_x_cg`: Conjugate Gradient Descent ihvp algorithm function with fixed x.
- `ihvp_arnoldi`: Arnoldi Iteration ihvp algorithm function.
- `ihvp_at_x_arnoldi`: Arnoldi Iteration ihvp algorithm function with fixed x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ClassVar, Generator, List, Optional, Tuple, Union


import warnings
from functools import wraps

import torch
from torch import Tensor
from torch.func import grad, hessian, jvp, vjp, vmap


def hvp(
    func: Callable,
    argnums: int = 0,
    mode: str = "rev-rev",
    regularization: float = 0.0,
) -> Callable:
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


def hvp_at_x(
    func: Callable,
    x: Tuple[torch.Tensor, ...],
    argnums: int = 0,
    mode: str = "rev-rev",
    regularization: float = 0.0,
) -> Callable:
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


def ihvp_explicit(
    func: Callable,
    argnums: int = 0,
    regularization: float = 0.0,
) -> Callable:
    """IHVP via explicit Hessian calculation.

    IHVP stands for inverse-hessian-vector product. For a given function
    `func`, this method first calculates the Hessian matrix explicitly
    and then wraps the Hessian in a function that uses `torch.linalg.solve` to
    calculate the IHVP for any given vector.

    Args:
        func (Callable): A function taking one or more arguments and returning
            a single-element Tensor. The Hessian will be calculated based on
            this function.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse hessian with respect to.
        regularization (float): A float default to 0.0. Specifies the regularization
            term to be added to the Hessian matrix. This is useful when the Hessian
            matrix is singular or ill-conditioned. The regularization term is
            `regularization * I`, where `I` is the identity matrix directly added
            to the Hessian matrix.

    Returns:
        A function that takes a tuple of Tensor `x` and a vector `v` and returns
        the IHVP of the Hessian of `func` and `v`.
    """
    hessian_func = hessian(func, argnums=argnums)

    def _ihvp_explicit_func(x: Tuple[torch.Tensor, ...], v: Tensor) -> Tensor:
        """The IHVP function using explicit hessian.

        Args:
            x (Tuple[torch.Tensor, ...]): The function will computed the
                inverse hessian matrix with respect to these arguments.
            v (Tensor): The vector to be produced on the inverse hessian matrix.

        Returns:
            The IHVP value.
        """
        hessian_tensor = hessian_func(*x)
        return torch.linalg.solve(
            hessian_tensor
            + torch.eye(hessian_tensor.shape[0]).to(v.device) * regularization,
            v.T,
        ).T

    return _ihvp_explicit_func


def ihvp_at_x_explicit(
    func: Callable,
    *x,
    argnums: Union[int, Tuple[int, ...]] = 0,
    regularization: float = 0.0,
) -> Callable:
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
        return torch.linalg.solve(
            hessian_tensor + torch.eye(hessian_tensor.shape[0]) * regularization,
            v.T,
        ).T

    return _ihvp_at_x_explicit_func


def ihvp_cg(
    func: Callable,
    argnums: int = 0,
    max_iter: int = 10,
    tol: float = 1e-7,
    mode: str = "rev-rev",
    regularization: float = 0.0,
) -> Callable:
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
        max_iter (int): An integer default 10. Specifies the maximum iteration
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
        return ihvp_at_x_cg(
            func,
            *x,
            argnums=argnums,
            max_iter=max_iter,
            tol=tol,
            mode=mode,
            regularization=regularization,
        )(v)

    return _ihvp_cg_func


def ihvp_at_x_cg(
    func: Callable,
    *x,
    argnums: int = 0,
    max_iter: int = 10,
    tol: float = 1e-7,  # noqa: ARG001
    mode: str = "rev-rev",
    regularization: float = 0.0,
) -> Callable:
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
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse hessian with respect to.
        max_iter (int): An integer default 10. Specifies the maximum iteration
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
    hvp_at_x_func = hvp_at_x(
        func,
        x=(*x,),
        argnums=argnums,
        mode=mode,
        regularization=regularization,
    )

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

        def _cg(v_i: Tensor) -> Tensor:
            x_pre = torch.clone(v_i)
            ihvp_res = x_pre
            g_pre = v_i - hvp_at_x_func(ihvp_res)
            d = d_pre = g_pre

            for _ in range(max_iter):
                # TODO: add tol for residual, it has not been supported by vmap.
                # https://pytorch.org/docs/main/generated/torch.cond.html#torch.cond
                # Still in prototype stage.

                ad = hvp_at_x_func(d)
                alpha = torch.dot(g_pre, d_pre) / torch.dot(d, ad)
                ihvp_res += alpha * d
                g = g_pre - alpha * ad

                beta = torch.dot(g, g) / torch.dot(g_pre, g_pre)

                g_pre = d_pre = g
                d = g + beta * d
            return ihvp_res

        return vmap(_cg)(v)

    return _ihvp_cg_func


def ihvp_arnoldi(
    func: Callable,
    argnums: int = 0,
    max_iter: int = 100,
    tol: float = 1e-7,
    mode: str = "rev-fwd",
    regularization: float = 0.0,
) -> Callable:
    """Arnoldi Iteration ihvp algorithm function.

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    Arnoldi Iteration builds an approximately H-invariant subspace by constructing
    the n-th order Krylov subspace and builds an orthonormal basis for it.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The hessian will
            be calculated on this function.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse hessian with respect to.
        max_iter (int): An integer default 100. Specifies the maximum iteration
            to calculate the ihvp through Arnoldi Iteration.
        tol (float): A float default to 1e-7. Specifies the break condition that
            decide if the algorithm has converged. If the torch.norm of current
            basis vector is less than tol, then the arnoldi_iter algorithm is truncated.
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

    def _ihvp_arnoldi_func(x: Tuple[torch.Tensor, ...], v: Tensor) -> Tensor:
        """The IHVP function using Arnoldi Iteration.

        Args:
            x (Tuple[torch.Tensor, ...]): The function will computed the
                inverse hessian matrix with respect to these arguments.
            v (Tensor): The vector to be produced on the inverse hessian matrix.

        Returns:
            The IHVP value.
        """
        return ihvp_at_x_arnoldi(
            func,
            *x,
            argnums=argnums,
            max_iter=max_iter,
            tol=tol,
            mode=mode,
            regularization=regularization,
        )(v)

    return _ihvp_arnoldi_func


def ihvp_at_x_arnoldi(
    func: Callable,
    *x,
    argnums: int = 0,
    max_iter: int = 100,
    top_k: int = 100,
    norm_constant: float = 1.0,
    tol: float = 1e-7,
    mode: str = "rev-fwd",
    regularization: float = 0.0,
) -> Callable:
    """Arnoldi Iteration ihvp algorithm function (with fixed x).

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    Arnoldi Iteration builds an approximately H-invariant subspace by constructing
    the n-th order Krylov subspace and builds an orthonormal basis for it.


    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The hessian will
            be calculated on this function.
        *x: List of arguments for `func`.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse hessian with respect to.
        max_iter (int): An integer default to 100. Specifies the maximum iteration
            to calculate the ihvp through Arnoldi Iteration.
        top_k (int): An integer default to 100. Specifies how many eigenvalues and
            eigenvectors to distill.
        norm_constant (float): A float default to 1.0. Specifies a constant value
            for the norm of each projection. In some situations (e.g. with a large
            numbers of parameters) it might be advisable to set norm_constant > 1
            to avoid dividing projection components by a large normalization factor.
        tol (float): A float default to 1e-7. Specifies the break condition that
            decide if the algorithm has converged. If the torch.norm of current
            basis vector is less than tol, then the arnoldi_iter algorithm is truncated.
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
    # algorithm refer to
    # https://github.com/google-research/jax-influence/blob/main/jax_influence/arnoldi.py

    hvp_at_x_func = hvp_at_x(
        func,
        x=(*x,),
        argnums=argnums,
        mode=mode,
        regularization=regularization,
    )

    def _arnoldi_iter(
        hvp_func: Callable,
        start_vec: Tensor,
        n_iters: int,
        norm_constant: float,
        tol: float,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        """Applies Arnoldi's algorithm.

        Args:
            hvp_func (Callable): A function that computes hvp.
            start_vec (Tensor): A random normalized vector for initialization.
            n_iters (int): The number of iteration.
            norm_constant (float): The norm normalization for each projection.
            tol (float): A tolerance value used to terminate iteration early.
            device (str): The device to run the algorithm. Defaults to "cpu".

        Returns:
            The result of the Arnoldi Iteration, containing a Hessenberg
            matrix H' approximating the Hessian matrix on its Krylov subspace K,
            and the projections onto K. If H is Hermitian,
            H' will be a tridiagonal matrix (up to numerical errors).
        """
        n_iters = min(start_vec.shape[0] + 1, n_iters)

        proj = []
        appr_mat = torch.zeros((n_iters, n_iters - 1)).to(device)

        start_vec /= torch.norm(start_vec)
        proj.append(start_vec)

        for n in range(n_iters - 1):
            h_vec = hvp_func(proj[n])

            for j, proj_vec in enumerate(proj):
                appr_mat[j][n] = torch.dot(h_vec, proj_vec) / norm_constant**2
                h_vec -= appr_mat[j][n] * proj_vec

            new_norm = torch.norm(h_vec)
            if new_norm < tol:
                appr_mat[n + 1][n] = 0
                proj.append(h_vec)
                appr_mat = appr_mat[: n + 2, : n + 1]
                break

            appr_mat[n + 1][n] = new_norm / norm_constant
            h_vec *= 1.0 / appr_mat[n + 1][n]
            proj.append(h_vec)

        return appr_mat, torch.stack(proj, dim=0)

    def _distill(
        appr_mat: Tensor,
        proj: Tensor,
        top_k: int,
        *,
        force_hermitian: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Distills result of Arnoldi iteration to top_k eigenvalues and eigenvectors.

        Args:
            appr_mat (Tensor): The first result from arnoldi_iter. This will be a
                Hessenberg matrix H' approximating the Hessian H.
            proj (Tensor): The second result from arnoldi_iter. This will be the
                projection vectors onto the Krylov subspace K of the Hessian H.
            top_k (int): Specfies how many eigenvalues and eigenvectors to distill.
            force_hermitian (bool): Whether to force the Hessian to Hermitian.
                Defaults to True.

        Returns:
            The distilled eigenvalues and eigenvectors.
        """
        appr_mat = appr_mat[:-1, :]

        if force_hermitian:
            appr_mat = torch.tril(appr_mat, diagonal=1)
            appr_mat = 0.5 * (appr_mat + appr_mat.T)
            eigvals, eigvecs = torch.linalg.eigh(appr_mat)
        else:
            eigvals, eigvecs = torch.linalg.eig(appr_mat)

        idx = torch.argsort(torch.abs(eigvals))
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        reduced_projections = torch.matmul(eigvecs[:, -top_k:].T, proj[:-1])

        return eigvals[-top_k:], reduced_projections

    def _ihvp_at_x_arnoldi(v: Tensor) -> Tensor:
        """The IHVP function using Arnoldi Iteration.

        Args:
            v (Tensor): The vector to be produced on the inverse hessian matrix.

        Returns:
            The IHVP value.
        """
        # algorithm refer to
        # https://github.com/google-research/jax-influence/blob/main/jax_influence/arnoldi.py

        if v.ndim == 1:
            v = v.unsqueeze(0)
        v0 = torch.rand(v.shape[1]).to(v.device)

        appr_mat, proj = _arnoldi_iter(
            hvp_at_x_func,
            v0,
            max_iter,
            norm_constant,
            tol,
            v.device,
        )
        eigvals, eigvecs = _distill(appr_mat, proj, top_k)

        return ((v @ eigvecs.T) * 1.0 / eigvals.unsqueeze(0)) @ eigvecs

    return _ihvp_at_x_arnoldi


def _check_input_size(*x, in_dims: Optional[Tuple] = None) -> int:
    """Check and return the size of input data.

    Args:
        *x: List of arguments to check. Each argument shoule be either:
            1. A tensor with a batch size dimension. Each data point i
            will take the i-th element along this dimension.
            2. A tensor without a batch size dimension. Each data point will
            share this tensor.
        in_dims (Optional[Tuple]): A tuple with the same shape as *x, indicating
            which dimension should be considered as batch size dimension. Take the
            first dimension as batch size dimension by default.

    Returns:
        An integer indicating the number of input data.

    Raises:
        IHVPUsageError: if the input size is ambiguous or mismatches.
    """
    if in_dims is None:
        in_dims = (0,) * len(x)

    if len(in_dims) != len(x):
        message = "Length of `in_dim` mismatches length of `*x`."
        raise IHVPUsageError(message)

    # Check batch size mismatch
    batch_size = None
    for i, (x_in, dim) in enumerate(zip(x, in_dims)):
        if dim is None:
            continue

        if batch_size is None:
            batch_size = x_in.shape[dim]
        elif batch_size != x_in.shape[dim]:
            message = (f"Input batch size mismatch! Expected {batch_size}, "
                       f"found {x_in.shape[dim]} for input tensor {i}.")
            raise IHVPUsageError(message)

    if batch_size is None:
        # For EK-FAC, we allow the sample size to be 1
        batch_size = 1

    return batch_size


def _sample_random_batch(*x,
                         num_samples: int,
                         in_dims: Optional[Tuple] = None,
                         batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
    """Randomly sample a batch of `batch_size` from the input data, without replacement.

    Args:
        *x: List of arguments to check. Each argument shoule be either:
            1. A tensor with a batch size dimension. Each data point i
            will take the i-th element along this dimension.
            2. A tensor without a batch size dimension. Each data point will
            share this tensor.
        num_samples (int): An integer, indicating the total number of samples.
        batch_size (int): An integer default to 1, indicating the batch size.
        in_dims (Optional[Tuple]): A tuple with the same shape as *x, indicating
            which dimension should be considered as batch size dimension. Take the
            first dimension as batch size dimension by default.

    Returns:
        An tuple of tensors corresponding to a batch of input data.
    """
    if in_dims is None:
        in_dims = (0,) * len(x)

    # Randomly sample and collate a batch
    sampled_indices = torch.randperm(num_samples)[:batch_size]

    return tuple(
        x_in.index_select(dim, sampled_indices)
        if dim is not None else x_in
        for x_in, dim in zip(x, in_dims)
    )


def ihvp_lissa(func: Callable,
               argnums: int = 0,
               batch_size: int = 1,
               num_repeat: int = 1,
               recursion_depth: int = 5000,
               damping: int = 0.0,
               scaling: int = 50.0,
               mode: str = "rev-rev") -> Callable:
    """IHVP via LiSSA algorithm.

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
        batch_size (int): An integer default to 1. Specifies the batch size used
            for LiSSA inner loop update.
        num_repeat (int): An integer default to 1. Specifies the number of samples
            of the hvp approximation to average on.
        recursion_depth (int): A integer default to 5000. Specifies the number of
            recursions used to estimate each ihvp sample.
        damping (int): Damping factor used for non-convexity in LiSSA ihvp calculation.
        scaling (int): Scaling factor used for convergence in LiSSA ihvp calculation.
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
    hvp_func = hvp(func, argnums=argnums, mode=mode)

    def _ihvp_lissa_func(x: Tuple[torch.Tensor, ...],
                         v: Tensor,
                         in_dims: Optional[Tuple] = None) -> Tensor:
        """The IHVP function via LiSSA algorithm.

        Args:
            x (Tuple[torch.Tensor, ...]): The function will computed the
                inverse hessian matrix with respect to these arguments.
            v (Tensor): The vector to be produced on the inverse hessian matrix.
            in_dims (Optional[Tuple]): A tuple with the same shape as *x. Indicating
                which dimension should be considered as batch size dimension.

        Returns:
            The IHVP value.
        """
        num_samples = _check_input_size(*x, in_dims=in_dims)
        if v.ndim == 1:
            v = v.unsqueeze(0)

        def _lissa_loop(vec: torch.Tensor) -> torch.Tensor:
            ihvp_estimations = []
            for _ in range(num_repeat):
                curr_estimate = vec.detach().clone()  # No gradient on v
                for _ in range(recursion_depth):
                    sampled_input = _sample_random_batch(*x,
                                                         batch_size=batch_size,
                                                         num_samples=num_samples,
                                                         in_dims=in_dims)
                    hvp = hvp_func(sampled_input, curr_estimate)
                    curr_estimate = (vec
                                     + (1 - damping) * curr_estimate
                                     - hvp / scaling)

                ihvp_estimations.append(curr_estimate / scaling)

            return torch.mean(torch.stack(ihvp_estimations), dim=0)

        return torch.vmap(_lissa_loop, randomness="different")(v)

    return _ihvp_lissa_func


def ihvp_at_x_lissa(func: Callable,
                    *x,
                    in_dims: Optional[Tuple] = None,
                    argnums: int = 0,
                    batch_size: int = 1,
                    num_repeat: int = 1,
                    recursion_depth: int = 5000,
                    damping: int = 0.0,
                    scaling: int = 50.0,
                    mode: str = "rev-rev") -> Callable:
    """IHVP with fixed func inputs via LiSSA algorithm.

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    LiSSA algorithm approximates the ihvp function by averaging multiple samples.
    The samples are estimated by recursion based on Taylor expansion.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The hessian will
            be estimated on this function.
        *x: List of arguments for `func`.
        in_dims (Optional[Tuple]): A tuple with the same shape as *x, indicating
            which dimension should be considered as batch size dimension. Take the
            first dimension as batch size dimension by default.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse hessian with respect to.
        batch_size (int): An integer default to 1. Specifies the batch size used
            for LiSSA inner loop update.
        num_repeat (int): An integer default to 1. Specifies the number of samples
            of the hvp approximation to average on.
        recursion_depth (int): A integer default to 5000. Specifies the number of
            recursions used to estimate each ihvp sample.
        damping (int): Damping factor used for non-convexity in LiSSA ihvp calculation.
        scaling (int): Scaling factor used for convergence in LiSSA ihvp calculation.
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

    def _ihvp_at_x_lissa_func(v: Tensor) -> Tensor:
        """The IHVP function with fixed func inputs using LiSSA.

        Args:
            v (Tensor): The vector to be produced on the inverse hessian matrix.

        Returns:
            The IHVP value.
        """
        ihvp_lissa_func = ihvp_lissa(func,
                                     argnums,
                                     num_repeat,
                                     batch_size,
                                     recursion_depth,
                                     damping,
                                     scaling,
                                     mode)
        return ihvp_lissa_func(x, v, in_dims=in_dims)

    return _ihvp_at_x_lissa_func


EKFAC_CACHE_KEY = "__cache"


def manual_cache_forward(forward_func: Callable) -> Callable:
    """Decorator for caching the input, output and gradient information for a module.

    Manually rewrite the forward function to collect variables you are interested in.

    Args:
        forward_func: the forward function to wrap up.

    Returns:
        The forward function with caching.

    Examples:
        @manual
        def custom_forward_method(self, hidden_states):
            if not hasattr(self, EKFAC_CACHE_KEY):
                # Normal forward pass
                hidden_states = hidden_states.view(hidden_states.shape[0], -1)
                return self.linear(hidden_states)

            # Forward pass with caching i/o variables
            cache = getattr(self, EKFAC_CACHE_KEY)
            x1 = hidden_states.view(hidden_states.shape[0], -1)
            y1 = self.linear(x1)
            cache.input_hidden_pairs.append((x1, y1))
            return y1
    """
    @wraps(forward_func)
    def cached_forward(self: torch.nn.Module, *args, **kwrds) -> torch.Tensor:
        if not hasattr(self, EKFAC_CACHE_KEY):
            return forward_func(self, *args, **kwrds)
        cache = getattr(self, EKFAC_CACHE_KEY)
        cache.clear()
        outputs = forward_func(self, *args, **kwrds)
        cache.check_type()
        cache.retain_grad()
        return outputs
    return cached_forward


class MLPCache:
    """Cache of input and output variables in a MLP layer."""
    input_hidden_pairs: ClassVar[List[Tuple[torch.Tensor, ...]]] = []

    def clear(self) -> None:
        """Reset the variables in the cache."""
        self.input_hidden_pairs = []

    def zero_grad(self) -> None:
        """Zero out the gradients of the variables in the cache."""
        for inputs, hiddens in self.input_hidden_pairs:
            inputs.grad = None
            hiddens.grad = None

    def retain_grad(self) -> None:
        """Ensure the gradients of the tensors are retained after backward pass."""
        for inputs, hiddens in self.input_hidden_pairs:
            if inputs.requires_grad:
                inputs.retain_grad()
            if hiddens.requires_grad:
                hiddens.retain_grad()

    def check_type(self) -> None:
        """Ensure the correctness of types of the cached variables.

        Raises:
            IHVPUsageError: if any type of cached varibales is not torch.Tensor.
        """
        for inputs, hiddens in self.input_hidden_pairs:
            if not (isinstance(inputs, torch.Tensor) and
                    isinstance(hiddens, torch.Tensor)):
                message = ("Incorrect type of variable is cached in `MLPCache`. "
                           "Only `torch.Tensor` is supported.")
                raise IHVPUsageError(message)


def _random_batch_iterator(*x,
                           num_samples: int,
                           in_dims: Optional[Tuple] = None,
                           batch_size: int = 1,
                           ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
    """Randomly sample a batch of `batch_size` from the input data, without replacement.

       This iterator basically does the same thing as `torch.utils.data.DataLoader`,
       with the default collation function.

    Args:
        *x: List of arguments to check. Each argument shoule be either:
            1. A tensor with a batch size dimension. Each data point i
            will take the i-th element along this dimension.
            2. A tensor without a batch size dimension. Each data point will
            share this tensor.
        num_samples (int): An integer, indicating the total number of samples.
        batch_size (int): An integer default to 1, indicating the batch size.
        in_dims (Optional[Tuple]): A tuple with the same shape as *x, indicating
            which dimension should be considered as batch size dimension. Take the
            first dimension as batch size dimension by default.

    Yields:
        An tuple of tensors corresponding to a batch of input data.
            The last batch of input data will NOT be discarded.
    """
    if batch_size > num_samples:
        batch_size = num_samples
        message = ("`batch_size` is larger than total number of samples. "
                   "Use `num_Samples` instead.")
        warnings.warn(message, stacklevel=2)

    if in_dims is None:
        in_dims = (0,) * len(x)

    random_perm = torch.randperm(num_samples)

    for i in range((num_samples + batch_size - 1) // batch_size):
        # Randomly sample and collate a batch
        begin_idx = i * batch_size
        end_idx = min(num_samples, (i + 1) * batch_size)
        sampled_indices = random_perm[begin_idx: end_idx]
        yield tuple(
            x_in.index_select(dim, sampled_indices.to(x_in.device))
            if dim is not None else x_in
            for x_in, dim in zip(x, in_dims)
        )


def _estimate_covariance(curr_estimate: List[List[Tuple[torch.Tensor]]],
                         mlp_cache: List[MLPCache],
                         total_samples: int,
                         mask: torch.Tensor) -> List[List[Tuple[torch.Tensor]]]:
    """Estimate the 'covariance' matrices S and A in EK-FAC ihvp.

    Args:
        curr_estimate (List[List[Tuple[torch.Tensor]]]): A list of lists of tuples
            of tensors, storing the running estimation of the layer-wise covariances.
        mlp_cache (List[MLPCache]): A list of `MLPCache` passed to the main
            EK-FAC function.
        total_samples (int): An integer indicating the number of total valid
            samples in the current batch.
        mask (torch.Tensor): A tensor of shape (batch_size, t), where 1's
            indicate that the ihvp will be estimated on these input positions and
            0's indicate that these positions are irrelevant (e.g. padding tokens).

    Returns:
        A list of lists of tuples of tensors, storing the updated running covariances.
    """
    batch_samples = int(mask.sum())
    for cache, layer_cov in zip(mlp_cache, curr_estimate):
        for idx, (a_prev, s_curr) in enumerate(cache.input_hidden_pairs):
            a_prev_masked = a_prev * mask[..., None].to(a_prev.device)
            # Calculate batch covariance matrix for A
            a_prev_reshaped = a_prev_masked.view(-1, a_prev.size(-1))
            batch_cov_a = a_prev_reshaped.transpose(0, 1) @ a_prev_reshaped
            batch_cov_a /= batch_samples

            # Calculate batch covariance matrix for S
            ds_curr = s_curr.grad

            ds_curr_reshaped = ds_curr.view(-1, s_curr.size(-1))
            batch_cov_s = ds_curr_reshaped.transpose(0, 1) @ ds_curr_reshaped
            batch_cov_s /= batch_samples

            # Update the running covariance matrices for A and S
            if idx <= len(layer_cov):
                # First time initializartion
                layer_cov.append((batch_cov_a, batch_cov_s))
            else:
                old_weight = total_samples / (total_samples + batch_samples)
                new_weight = batch_samples / (total_samples + batch_samples)
                layer_cov[idx][0] = (old_weight * layer_cov[idx][0] +
                                     new_weight * batch_cov_a)
                layer_cov[idx][1] = (old_weight * layer_cov[idx][1] +
                                     new_weight * batch_cov_s)

    return curr_estimate


def _estimate_lambda(curr_estimate: List[List[torch.Tensor]],
                     mlp_cache: List[MLPCache],
                     cached_q: List[List[Tuple[torch.Tensor]]],
                     total_samples: int,
                     mask: torch.Tensor,
                     max_steps_for_vec: int = 10) -> List[List[torch.Tensor]]:
    """Estimate the corrected eigenvalues in EK-FAC ihvp.

    Args:
        curr_estimate (List[List[torch.Tensor]]): A list of lists of tensors,
            storing the running estimation of the layer-wise lambdas.
        mlp_cache (List[MLPCache]): A list of `MLPCache` passed to the main
            EK-FAC function.
        cached_q (List[List[Tuple[torch.Tensor]]]): A list of lists of tuples
            of tensors, storing the layer-wise eigenvector matrices calculated
            in the EK-FAC main function.
        total_samples (int): An integer indicating the number of total valid
            samples in the current batch.
        mask (torch.Tensor): A tensor of shape (batch_size, t), where 1's
            indicate that the ihvp will be estimated on these input positions and
            0's indicate that these positions are irrelevant (e.g. padding tokens).
        max_steps_for_vec (int): An integer default to 10. Controls the maximum
            number of input steps that is allowed for vectorized calculation of
            `dtheta`.

    Returns:
        A list of lists of tensors, storing the updated running lambdas.
    """
    for cache, layer_lambda, layer_q in zip(mlp_cache, curr_estimate, cached_q):
        for idx, ((a_prev, s_curr), (q_a, q_s)) in enumerate(
            zip(cache.input_hidden_pairs, layer_q)):
            a_prev_masked = a_prev * mask[..., None].to(a_prev.device)

            # Uniformly reshape the tensors into (batch_size, t, ...)
            # The t here is the sequence length or time steps for sequential input
            # t = 1 if the given input is not sequential
            ds_curr = s_curr.grad

            if a_prev.ndim == 2:  # noqa: PLR2004
                a_prev_reshaped = a_prev_masked.unsqueeze(1)
                ds_curr_reshaped = ds_curr.unsqueeze(1)
            else:
                a_prev_reshaped = a_prev_masked
                ds_curr_reshaped = ds_curr

            batch_samples = a_prev_reshaped.shape[0]
            timesteps = a_prev_reshaped.shape[1]  # the value of t
            if timesteps <= max_steps_for_vec:
                # Vectorized calculation of dtheta
                batch_dtheta = (ds_curr_reshaped.unsqueeze(-1) @
                                a_prev_reshaped.unsqueeze(2)).sum(axis=1)
            else:
                # Memory efficient calculation of dtheta
                batch_dtheta = torch.zeros(batch_samples,
                                           ds_curr_reshaped.shape[-1],
                                           a_prev_reshaped.shape[-1],
                                           device=ds_curr.device)

                for ts in range(timesteps):
                    batch_dtheta += (ds_curr_reshaped[:, ts, :, None] @
                                     a_prev_reshaped[:, ts, None, :])

            # An equivalent way to calculate lambda's. Please refer to
            # https://math.stackexchange.com/questions/1879933/vector-multiplication-with-multiple-kronecker-products
            batch_lambda = torch.square(q_s @ batch_dtheta @ q_a.T).mean(axis=0)

            # Update the running eigenvalue estimation
            if idx <= len(layer_lambda):
                # First time initializartion
                layer_lambda.append(batch_lambda)
            else:
                old_weight = total_samples / (total_samples + batch_samples)
                new_weight = batch_samples / (total_samples + batch_samples)
                layer_lambda[idx] = (old_weight * layer_lambda[idx] +
                                     new_weight * batch_lambda)

    return curr_estimate


def ihvp_at_x_ekfac(func: Callable,
                    *x,
                    in_dims: Optional[Tuple] = None,
                    batch_size: int = 1,
                    max_iter: Optional[int] = None,
                    mlp_cache: Union[MLPCache, List[MLPCache]],
                    damping: float = 0.0) -> Callable:
    """IHVP via EK-FAC algorithm.

    Standing for the inverse-hessian-vector product, returns a function that,
    when given vectors, computes the product of inverse-hessian and vector.

    EK-FAC algorithm provides layer-wise approximation for the ihvp function.
    The samples are estimated based on Gauss-Newton Hessian.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return the following,
            - losses: a tensor of shape (batch_size,).
            - mask (optional): a tensor of shape (batch_size, t), where 1's
                               indicate that the ihvp will be estimated on these
                               input positions and 0's indicate that these positions
                               are irrelevant (e.g. padding tokens).
            t is the number of steps, or sequence length of the input data. If the
            input data are non-sequential, t should be set to 1.
            The hessian will be estimated on this function.
        *x: List of arguments for `func`.
        in_dims (Tuple, optional): A tuple with the same shape as *x, indicating
            which dimension should be considered as batch size dimension. Take the
            first dimension as batch size dimension by default.
        batch_size (int): An integer default to 1, indicating the batch size used for
            estimating the covariance matrices and lambdas.
        max_iter (int, optional): An integer indicating the maximum number of
            batches that will be used for estimating the the covariance matrices and
            lambdas.
        mlp_cache (Union[MLPCache, List[MLPCache]]): A single or list of registered
            caches, used to record the input and hidden vectors as well as their
            relevant gradients during the forward and backward calls of `func`.
        damping: Damping factor used for non-convexity in EK-FAC ihvp calculation.

    Returns:
        A function that takes  a tuple of Tensor `x` and a nested structure of
        vector `v` and returns the IHVP of the Hessian of `func` and `v`.
    """
    num_samples = _check_input_size(*x, in_dims=in_dims)
    if not isinstance(mlp_cache, list):
        mlp_cache = [mlp_cache]

    # 1. Use random batch to estimate covariance matrices S and A
    dataloader = _random_batch_iterator(*x,
                                        num_samples=num_samples,
                                        in_dims=in_dims,
                                        batch_size=batch_size)

    cov_matrices = [[] for _ in range(len(mlp_cache))]
    total_samples = 0  # record total number of samples
    for i, batch in enumerate(dataloader):
        # Forward pass
        func_output = func(*batch)
        losses, mask = (func_output if isinstance(func_output, tuple)
                        else func_output, torch.tensor(1.))

        for loss in losses:
            # Backward pass
            loss.backward(retain_graph=True)

        with torch.no_grad():
            # Estimate covariance
            cov_matrices = _estimate_covariance(cov_matrices,
                                                mlp_cache,
                                                total_samples,
                                                mask)

        total_samples += int(mask.sum())
        if max_iter is not None and i == max_iter - 1:
            break

    # 2. Calculate the eigenvalue decomposition of S and A
    cached_q = [[] for _ in range(len(mlp_cache))]
    for layer_q, layer_cov in zip(cached_q, cov_matrices):
        for cov_a, cov_s in layer_cov:
            _, q_a = torch.linalg.eigh(cov_a, UPLO="U")
            _, q_s = torch.linalg.eigh(cov_s, UPLO="U")
            layer_q.append((q_a, q_s))

    # 3. Use random batch for eigenvalue correction
    dataloader = _random_batch_iterator(*x,
                                        num_samples=num_samples,
                                        in_dims=in_dims,
                                        batch_size=batch_size)
    cached_lambdas = [[] for _ in range(len(mlp_cache))]
    total_samples = 0  # record total number of samples
    for i, batch in enumerate(dataloader):
        # Forward pass
        func_output = func(*batch)
        losses, mask = (func_output if isinstance(func_output, tuple)
                        else func_output, torch.tensor(1.))

        for loss in losses:
            # Backward pass
            loss.backward(retain_graph=True)

        with torch.no_grad():
            # Update lambdas
            cached_lambdas = _estimate_lambda(cached_lambdas,
                                              mlp_cache,
                                              cached_q,
                                              total_samples,
                                              mask)
        total_samples += len(losses)
        if max_iter is not None and i == max_iter - 1:
            break

    # Clear unused data from cache
    del cov_matrices

    def _ihvp_at_x_ekfac_func(v: List[List[torch.Tensor]]) -> torch.Tensor:
        """The IHVP function with fixed func inputs using EK-FAC.

        Args:
            v (List[List[torch.Tensor]]): A list of tensors to be produced on the
                inverse hessian matrix, where each element should have the compatible
                shape (out_dim, in_dim) with the cached layers in `mlp_cache`.

        Returns:
            The IHVP value.
        """
        ihvp = [[] for _ in range(len(cached_q))]
        for layer_ihvp, layer_v, layer_lambda, layer_q in zip(ihvp, v,
                                                              cached_lambdas,
                                                              cached_q):
            for _v, _lambda, (q_a, q_s) in zip(layer_v, layer_lambda, layer_q):
                _ihvp = q_s.T @ (
                    (q_s @ _v @ q_a.T) /
                    (_lambda + damping)
                    ) @ q_a
                layer_ihvp.append(_ihvp)
        return ihvp

    return _ihvp_at_x_ekfac_func


class IHVPUsageError(Exception):
    """The usage exception class for ihvp module."""
