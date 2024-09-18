"""IFVP calculation functions.

IFVP (inverse FIM-vector product)
FIM (Fisher Information Matrix)

This module contains:
- `ifvp_explicit`: IFVP via explicit FIM calculation.
- `ifvp_at_x_explicit`: IFVP via explicit FIM calculation (with fixed x).
- `ifvp_at_x_ekfac`: EK-FAC IFVP algorithm function (with fixed x).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ClassVar, Dict, Generator, List, Optional, Tuple, Union

    from torch import Tensor


import warnings
from functools import wraps

import torch
from torch.func import grad


def ifvp_explicit(
    func: Callable,
    argnums: int = 0,
    regularization: float = 0.0,
) -> Callable:
    """IFVP via explicit FIM calculation.

    IFVP stands for inverse-FIM-vector product. For a given function
    `func`, this method first calculates the FIM explicitly
    and then wraps the FIM in a function that uses `torch.linalg.solve` to
    calculate the IFVP for any given vector.

    Args:
        func (Callable): A function taking one or more arguments and returning
            a single-element Tensor. The FIM will be calculated based on
            this function. Notably, this function should be negative log-likelihood
            (e.g., cross-entropy loss) for classification tasks. If you want to
            calculate the empirical FIM, you should use the ground truth label for
            the loss. If you want to calculate the true FIM, you should use the
            predicted label for the loss.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse FIM with respect to.
        regularization (float): A float default to 0.0. Specifies the regularization
            term to be added to the FIM. This is useful when the FIM
            is singular or ill-conditioned. The regularization term is
            `regularization * I`, where `I` is the identity matrix directly added
            to the FIM.

    Returns:
        A function that takes a tuple of Tensor `x` and a vector `v` and returns
        the product of the FIM of `func` and `v`.
    """
    grad_func = grad(func, argnums=argnums)

    def _ifvp_explicit_func(x: Tuple[torch.Tensor, ...], v: Tensor) -> Tensor:
        """The IFVP function using explicit FIM.

        Args:
            x (Tuple[torch.Tensor, ...]): The function will computed the
                inverse FIM with respect to these arguments.
            v (Tensor): The vector to be produced on the inverse FIM.

        Returns:
            The IFVP value.
        """
        grad_tensor = grad_func(*x)
        return torch.linalg.solve(
            grad_tensor @ grad_tensor.T
            + torch.eye(grad_tensor.shape[0]).to(v.device) * regularization,
            v.T,
        ).T

    return _ifvp_explicit_func


def ifvp_at_x_explicit(
    func: Callable,
    *x,
    argnums: Union[int, Tuple[int, ...]] = 0,
    regularization: float = 0.0,
) -> Callable:
    """IFVP via explicit FIM calculation.

    IFVP stands for inverse-FIM-vector product. For a given function
    `func`, this method first calculates the FIM explicitly
    and then wraps the FIM in a function that uses `torch.linalg.solve` to
    calculate the IFVP for any given vector.

    Args:
        func (Callable): A function taking one or more arguments and returning
            a single-element Tensor. The FIM will be calculated based on
            this function. Notably, this function should be negative log-likelihood
            (e.g., cross-entropy loss) for classification tasks. If you want to
            calculate the empirical FIM, you should use the ground truth label for
            the loss. If you want to calculate the true FIM, you should use the
            predicted label for the loss.
        *x: List of arguments for `func`.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse FIM with respect to.
        regularization (float): A float default to 0.0. Specifies the regularization
            term to be added to the FIM. This is useful when the FIM
            is singular or ill-conditioned. The regularization term is
            `regularization * I`, where `I` is the identity matrix directly added
            to the FIM.

    Returns:
        A function that takes a vector `v` and returns the IFVP of the Hessian
        of `func` and `v`.
    """
    grad_tensor = grad(func, argnums=argnums)(*x)
    fim = grad_tensor @ grad_tensor.T

    def _ifvp_at_x_explicit_func(v: Tensor) -> Tensor:
        """The IFVP function using explicit FIM.

        Args:
            v (Tensor): The vector to be produced on the inverse FIM.

        Returns:
            The IFVP value.
        """
        return torch.linalg.solve(
            fim + torch.eye(grad_tensor.shape[0]).to(v.device) * regularization,
            v.T,
        ).T

    return _ifvp_at_x_explicit_func


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
        IFVPUsageError: if the input size is ambiguous or mismatches.
    """
    if in_dims is None:
        in_dims = (0,) * len(x)

    if len(in_dims) != len(x):
        message = "Length of `in_dim` mismatches length of `*x`."
        raise IFVPUsageError(message)

    # Check batch size mismatch
    batch_size = None
    for i, (x_in, dim) in enumerate(zip(x, in_dims)):
        if dim is None:
            continue

        if batch_size is None:
            batch_size = x_in.shape[dim]
        elif batch_size != x_in.shape[dim]:
            message = (
                f"Input batch size mismatch! Expected {batch_size},"
                f"found {x_in.shape[dim]} for input tensor {i}."
            )
            raise IFVPUsageError(message)

    if batch_size is None:
        # For EK-FAC, we allow the sample size to be 1
        batch_size = 1

    return batch_size


EKFAC_CACHE_KEY = "__ekfac_cache"


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
            IFVPUsageError: if any type of cached varibales is not torch.Tensor.
        """
        for inputs, hiddens in self.input_hidden_pairs:
            if not (
                isinstance(inputs, torch.Tensor) and isinstance(hiddens, torch.Tensor)
            ):
                message = (
                    "Incorrect type of variable is cached in `MLPCache`. "
                    "Only `torch.Tensor` is supported."
                )
                raise IFVPUsageError(message)


def _random_batch_iterator(
    *x,
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
        message = (
            "`batch_size` is larger than total number of samples. "
            "Use `num_Samples` instead."
        )
        warnings.warn(message, stacklevel=2)

    if in_dims is None:
        in_dims = (0,) * len(x)

    random_perm = torch.randperm(num_samples)

    for i in range((num_samples + batch_size - 1) // batch_size):
        # Randomly sample and collate a batch
        begin_idx = i * batch_size
        end_idx = min(num_samples, (i + 1) * batch_size)
        sampled_indices = random_perm[begin_idx:end_idx]
        yield tuple(
            x_in.index_select(dim, sampled_indices.to(x_in.device))
            if dim is not None
            else x_in
            for x_in, dim in zip(x, in_dims)
        )


def _update_covariance(
    curr_estimate: Dict[str, Tuple[torch.tensor]],
    layer_cache: Dict[str, Tuple[torch.tensor]],
    total_samples: int,
    mask: torch.Tensor,
) -> Dict[str, Tuple[torch.tensor]]:
    """Update the running estimation of the 'covariance' matrices S and A in EK-FAC IFVP.

    Args:
        curr_estimate (List[List[Tuple[torch.Tensor]]]): A list of lists of tuples
            of tensors, storing the running estimation of the layer-wise covariances.
        mlp_cache (List[MLPCache]): A list of `MLPCache` passed to the main
            EK-FAC function.
        total_samples (int): An integer indicating the number of total valid
            samples in the current batch.
        mask (torch.Tensor): A tensor of shape (batch_size, t), where 1's
            indicate that the IFVP will be estimated on these input positions and
            0's indicate that these positions are irrelevant (e.g. padding tokens).

    Returns:
        A list of lists of tuples of tensors, storing the updated running covariances.
    """
    batch_samples = int(mask.sum())
    for layer_name, (a_prev, s_curr) in layer_cache.items():
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
        if layer_name in curr_estimate:
            curr_estimate[layer_name] = (batch_cov_a, batch_cov_s)
        else:
            # First time access
            old_weight = total_samples / (total_samples + batch_samples)
            new_weight = batch_samples / (total_samples + batch_samples)
            new_cov_a = old_weight * curr_estimate[layer_name][0] + new_weight * batch_cov_a
            new_cov_s = old_weight * curr_estimate[layer_name][1] + new_weight * batch_cov_s
            curr_estimate[layer_name] = (new_cov_a, new_cov_s)

    return curr_estimate


def _update_lambda(
    curr_estimate: List[List[torch.Tensor]],
    layer_cache: Dict[str, Tuple[torch.tensor]],
    cached_q: List[List[Tuple[torch.Tensor]]],
    total_samples: int,
    mask: torch.Tensor,
    max_steps_for_vec: int = 10,
) -> List[List[torch.Tensor]]:
    """Update the running estimation of the corrected eigenvalues in EK-FAC IFVP.

    Args:
        curr_estimate (List[List[torch.Tensor]]): A list of lists of tensors,
            storing the running estimation of the layer-wise lambdas. The list
            has the same length as `mlp_cache` in the main function, and each
            of the member has the same length as the list in the cache.
        mlp_cache (List[MLPCache]): A list of `MLPCache` passed to the main
            EK-FAC function.
        cached_q (List[List[Tuple[torch.Tensor]]]): A list of lists of tuples
            of tensors, storing the layer-wise eigenvector matrices calculated
            in the EK-FAC main function.
        total_samples (int): An integer indicating the number of total valid
            samples in the current batch.
        mask (torch.Tensor): A tensor of shape (batch_size, t), where 1's
            indicate that the IFVP will be estimated on these input positions and
            0's indicate that these positions are irrelevant (e.g. padding tokens).
            t is the number of steps, or sequence length of the input data. If the
            input data are non-sequential, t should be set to 1.
        max_steps_for_vec (int): An integer default to 10. Controls the maximum
            number of input steps that is allowed for vectorized calculation of
            `dtheta`.

    Returns:
        A list of lists of tensors, storing the updated running lambdas.
    """
    for layer_name, (a_prev, s_curr) in layer_cache.items():
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
            batch_dtheta = (
                ds_curr_reshaped.unsqueeze(-1) @ a_prev_reshaped.unsqueeze(2)
            ).sum(axis=1)
        else:
            # Memory efficient calculation of dtheta
            batch_dtheta = torch.zeros(
                batch_samples,
                ds_curr_reshaped.shape[-1],
                a_prev_reshaped.shape[-1],
                device=ds_curr.device,
            )

        # An equivalent way to calculate lambda's. Please refer to
        # https://math.stackexchange.com/questions/1879933/vector-multiplication-with-multiple-kronecker-products
        q_s, q_a = cached_q[layer_name]
        batch_lambda = torch.square(q_s @ batch_dtheta @ q_a.T).mean(axis=0)

        # Update the running eigenvalue estimation
        if layer_name in curr_estimate:
            # First time initializartion
            curr_estimate[layer_name] = batch_lambda
        else:
            old_weight = total_samples / (total_samples + batch_samples)
            new_weight = batch_samples / (total_samples + batch_samples)
            curr_estimate[layer_name] = (
                old_weight * curr_estimate[layer_name] + new_weight * batch_lambda
            )

    return curr_estimate


def estimate_covariance(
    func: Callable,
    dataloader: torch.utils.data.DataLoader,
    layer_cache: Dict[str, Tuple[torch.tensor]],
    max_iter: int,
) -> Dict[str, Tuple[torch.tensor]]:
    covariances = {}
    total_samples = 0  # record total number of samples
    for i, batch in enumerate(dataloader):
        # Forward pass
        func_output = func(*batch)
        losses, mask = (
            func_output if isinstance(func_output, tuple) else func_output,
            torch.tensor(1.0),
        )

        for loss in losses:
            # Backward pass
            loss.backward(retain_graph=True)

        with torch.no_grad():
            # Estimate covariance
            cov_matrices = _update_covariance(
                covariances,
                layer_cache,
                total_samples,
                mask,
            )

        total_samples += int(mask.sum())
        if i == max_iter - 1:
            break

    return covariances


def estimate_eigenvector(
    covariances: List[List[Tuple[torch.Tensor]]],
) -> List[List[Tuple[torch.Tensor]]]:
    cached_q = {}
    for layer_name, (cov_a, cov_s) in covariances:
        _, q_a = torch.linalg.eigh(cov_a, UPLO="U")
        _, q_s = torch.linalg.eigh(cov_s, UPLO="U")
        cached_q[layer_name] = (q_a, q_s)

    return cached_q


def estimate_lambda(
    func: Callable,
    dataloader: torch.utils.data.DataLoader,
    eigenvectors: List[List[Tuple[torch.Tensor]]],
    layer_cache: Dict[str, Tuple[torch.tensor]],
    max_iter: int,
) -> Dict[str, Tuple[torch.tensor]]:
    lambdas = {}
    total_samples = 0  # record total number of samples
    for i, batch in enumerate(dataloader):
        # Forward pass
        func_output = func(*batch)
        losses, mask = (
            func_output if isinstance(func_output, tuple) else func_output,
            torch.tensor(1.0),
        )

        for loss in losses:
            # Backward pass
            loss.backward(retain_graph=True)

        with torch.no_grad():
            # Update lambdas
            lambdas = _update_lambda(
                lambdas,
                layer_cache,
                eigenvectors,
                total_samples,
                mask,
            )
        total_samples += len(losses)
        if i == max_iter - 1:
            break

    return lambdas


def ifvp_at_x_ekfac(
    func: Callable,
    *x,
    mlp_cache: Union[MLPCache, List[MLPCache]],
    in_dims: Optional[Tuple] = None,
    batch_size: int = 1,
    max_iter: Optional[int] = None,
    damping: float = 0.0,
) -> Callable:
    """IFVP via EK-FAC algorithm.

    Standing for the inverse-FIM-vector product, returns a function that,
    when given vectors, computes the product of inverse-FIM and vector.

    EK-FAC algorithm provides layer-wise approximation for the IFVP function.
    The samples are estimated based on Gauss-Newton Hessian.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return the following,
            - losses: a tensor of shape (batch_size,).
            - mask (optional): a tensor of shape (batch_size, t), where 1's
                               indicate that the IFVP will be estimated on these
                               input positions and 0's indicate that these positions
                               are irrelevant (e.g. padding tokens).
            t is the number of steps, or sequence length of the input data. If the
            input data are non-sequential, t should be set to 1.
            The FIM will be estimated on this function.
        *x: List of arguments for `func`.
        mlp_cache (Union[MLPCache, List[MLPCache]]): A single or list of registered
            caches, used to record the input and hidden vectors as well as their
            relevant gradients during the forward and backward calls of `func`.
        in_dims (Tuple, optional): A tuple with the same shape as *x, indicating
            which dimension should be considered as batch size dimension. Take the
            first dimension as batch size dimension by default.
        batch_size (int): An integer default to 1, indicating the batch size used for
            estimating the covariance matrices and lambdas.
        max_iter (int, optional): An integer indicating the maximum number of
            batches that will be used for estimating the the covariance matrices and
            lambdas.
        damping: Damping factor used for non-convexity in EK-FAC IFVP calculation.

    Returns:
        A function that takes  a tuple of Tensor `x` and a nested structure of
        vector `v` and returns the IFVP of the Hessian of `func` and `v`.
    """
    num_samples = _check_input_size(*x, in_dims=in_dims)
    if not isinstance(mlp_cache, list):
        mlp_cache = [mlp_cache]

    if max_iter is None:
        max_iter = (num_samples + batch_size - 1) // batch_size

    # 1. Use random batch to estimate covariance matrices S and A
    dataloader = _random_batch_iterator(
        *x,
        num_samples=num_samples,
        in_dims=in_dims,
        batch_size=batch_size,
    )
    cov_matrices = estimate_covariance(func,
                                       dataloader,
                                       mlp_cache,
                                       max_iter)

    # 2. Calculate the eigenvalue decomposition of S and A
    cached_q = estimate_eigenvector(cov_matrices,
                                    mlp_cache)

    # 3. Use random batch for eigenvalue correction
    dataloader = _random_batch_iterator(
        *x,
        num_samples=num_samples,
        in_dims=in_dims,
        batch_size=batch_size,
    )
    cached_lambdas = estimate_lambda(func,
                                     dataloader,
                                     cached_q,
                                     mlp_cache,
                                     max_iter)

    # Clear unused data from cache
    del cov_matrices

    def _ifvp_at_x_ekfac_func(v: List[List[torch.Tensor]]) -> torch.Tensor:
        """The IFVP function with fixed func inputs using EK-FAC.

        Args:
            v (List[List[torch.Tensor]]): A list of tensors to be produced on the
                inverse FIM, where each element should have the compatible
                shape (out_dim, in_dim) with the cached layers in `mlp_cache`.

        Returns:
            The IFVP value.
        """
        ifvp = [[] for _ in range(len(cached_q))]
        for layer_ifvp, layer_v, layer_lambda, layer_q in zip(
            ifvp,
            v,
            cached_lambdas,
            cached_q,
        ):
            for _v, _lambda, (q_a, q_s) in zip(layer_v, layer_lambda, layer_q):
                _ifvp = q_s.T @ ((q_s @ _v @ q_a.T) / (_lambda + damping)) @ q_a
                layer_ifvp.append(_ifvp)
        return ifvp

    return _ifvp_at_x_ekfac_func

class IFVPUsageError(Exception):
    """The usage exception class for IFVP module."""
