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
    from typing import Dict, Generator, Optional, Tuple, Union

    from torch import Tensor


import warnings

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
    """Update the running estimation of the covariance matrices S and A in EK-FAC IFVP.

    Args:
        curr_estimate (List[List[Tuple[torch.Tensor]]]): A list of lists of tuples
            of tensors, storing the running estimation of the layer-wise covariances.
        layer_cache (Dict[str, Tuple[torch.tensor]]): A dict that caches a pair
            of (inputs, outputs) for each module during the forward process.
        total_samples (int): An integer indicating the number of total valid
            samples in all previous batchs.
        mask (torch.Tensor): A tensor of shape (batch_size, t), where 1's
            indicate that the IFVP will be estimated on these input positions and
            0's indicate that these positions are irrelevant (e.g. padding tokens).

    Returns:
        Dict[str, Tuple[torch.tensor]]: A dict of tuples of tensors, storing the
            updated running covariances.
    """
    batch_samples = int(mask.sum())
    for layer_name, (a_prev_raw, s_curr_raw) in layer_cache.items():
        # Uniformly reshape the tensors into (batch_size, t, ...)
        # The t here is the sequence length or time steps for sequential input
        # t = 1 if the given input is not sequential
        if a_prev_raw.ndim == 2:  # noqa: PLR2004
            a_prev = a_prev_raw.unsqueeze(1)

        a_prev_masked = a_prev * mask[..., None].to(a_prev.device)

        # Calculate batch covariance matrix for A
        a_prev_reshaped = a_prev_masked.view(-1, a_prev.size(-1))
        batch_cov_a = a_prev_reshaped.transpose(0, 1) @ a_prev_reshaped
        batch_cov_a /= batch_samples

        # Calculate batch covariance matrix for S
        ds_curr = s_curr_raw.grad

        ds_curr_reshaped = ds_curr.view(-1, s_curr_raw.size(-1))
        batch_cov_s = ds_curr_reshaped.transpose(0, 1) @ ds_curr_reshaped
        batch_cov_s /= batch_samples

        # Update the running covariance matrices for A and S
        if layer_name in curr_estimate:
            old_weight = total_samples / (total_samples + batch_samples)
            new_weight = batch_samples / (total_samples + batch_samples)
            new_cov_a = (old_weight * curr_estimate[layer_name][0] +
                         new_weight * batch_cov_a)
            new_cov_s = (old_weight * curr_estimate[layer_name][1] +
                         new_weight * batch_cov_s)
            curr_estimate[layer_name] = (new_cov_a, new_cov_s)
        else:
            # First time access
            curr_estimate[layer_name] = (batch_cov_a, batch_cov_s)

    return curr_estimate


def _update_lambda(
    curr_estimate: Dict[str, torch.tensor],
    layer_cache: Dict[str, Tuple[torch.tensor]],
    cached_q: Dict[str, Tuple[torch.tensor]],
    total_samples: int,
    mask: torch.Tensor,
    max_steps_for_vec: int = 10,
) -> Dict[str, torch.tensor]:
    """Update the running estimation of the corrected eigenvalues in EK-FAC IFVP.

    Args:
        curr_estimate (Dict[str, torch.tensor]): A list of lists of tensors,
            storing the running estimation of the layer-wise lambdas. The list
            has the same length as `mlp_cache` in the main function, and each
            of the member has the same length as the list in the cache.
        layer_cache (Dict[str, Tuple[torch.tensor]]): A dict that caches a pair
            of (inputs, outputs) for each module during the forward process.
        cached_q (Dict[str, Tuple[torch.tensor]]): A dict of tuples of tensors,
            storing the layer-wise eigenvector matrices.
        total_samples (int): An integer indicating the number of total valid
            samples in all previous batchs.
        mask (torch.Tensor): A tensor of shape (batch_size, t), where 1's
            indicate that the IFVP will be estimated on these input positions and
            0's indicate that these positions are irrelevant (e.g. padding tokens).
            t is the number of steps, or sequence length of the input data. If the
            input data are non-sequential, t should be set to 1.
        max_steps_for_vec (int): An integer default to 10. Controls the maximum
            number of input steps that is allowed for vectorized calculation of
            `dtheta`.

    Returns:
        Dict[str, torch.tensor]: A dict of tensors, storing the updated running lambdas.
    """
    for layer_name, (a_prev_raw, s_curr_raw) in layer_cache.items():
        # Uniformly reshape the tensors into (batch_size, t, ...)
        # The t here is the sequence length or time steps for sequential input
        # t = 1 if the given input is not sequential
        ds_curr = s_curr_raw.grad
        if a_prev_raw.ndim == 2:  # noqa: PLR2004
            a_prev = a_prev_raw.unsqueeze(1)
            ds_curr = ds_curr.unsqueeze(1)

        a_prev_masked = a_prev * mask[..., None].to(a_prev.device)

        batch_samples = a_prev_masked.shape[0]
        timesteps = a_prev_masked.shape[1]  # the value of t
        if timesteps <= max_steps_for_vec:
            # Vectorized calculation of dtheta
            batch_dtheta = (
                ds_curr.unsqueeze(-1) @ a_prev_masked.unsqueeze(2)
            ).sum(axis=1)
        else:
            # Memory efficient calculation of dtheta
            batch_dtheta = torch.zeros(
                batch_samples,
                ds_curr.shape[-1],
                a_prev_masked.shape[-1],
                device=ds_curr.device,
            )
            for ts in range(timesteps):
                batch_dtheta += (
                    ds_curr[:, ts, :, None]
                    @ a_prev_masked[:, ts, None, :]
                )

        # An equivalent way to calculate lambda's. Please refer to
        # https://math.stackexchange.com/questions/1879933/vector-multiplication-with-multiple-kronecker-products
        q_a, q_s = cached_q[layer_name]
        batch_lambda = torch.square(q_s @ batch_dtheta @ q_a.T).mean(axis=0)

        # Update the running eigenvalue estimation
        if layer_name in curr_estimate:
            old_weight = total_samples / (total_samples + batch_samples)
            new_weight = batch_samples / (total_samples + batch_samples)
            curr_estimate[layer_name] = (
                old_weight * curr_estimate[layer_name] + new_weight * batch_lambda
            )
        else:
            # First time initializartion
            curr_estimate[layer_name] = batch_lambda

    return curr_estimate


def estimate_covariance(
    func: Callable,
    dataloader: torch.utils.data.DataLoader,
    layer_cache: Dict[str, Tuple[torch.tensor]],
    max_iter: Optional[int] = None,
    device: Optional[str] = "cpu",
) -> Dict[str, Tuple[torch.tensor]]:
    """Estimate the 'covariance' matrices S and A in EK-FAC IFVP.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return the following,
            - loss: a single tensor of loss. Should be the mean loss by the
                    batch size.
            - mask (optional): a tensor of shape (batch_size, t), where 1's
                               indicate that the IFVP will be estimated on these
                               input positions and 0's indicate that these positions
                               are irrelevant (e.g. padding tokens).
            t is the number of steps, or sequence length of the input data. If the
            input data are non-sequential, t should be set to 1.
            The FIM will be estimated on this function.
        dataloader (torch.utils.data.DataLoader): The dataloader with full training
            samples for FIM estimation.
        layer_cache (Dict[str, Tuple[torch.tensor]]): A dict that caches a pair
            of (inputs, outputs) for each module during the forward process.
        max_iter (Optional[int]): An integer indicating the maximum number of
            batches that will be used for estimating the covariance matrices.
        device (Optional[str]): Device to run the attributor on. Default is "cpu".

    Returns:
        Dict[str, Tuple[torch.tensor]]: A dict that contains a pair of
            estimated covariance for each module.
    """
    if max_iter is None:
        max_iter = len(dataloader)

    covariances = {}
    total_samples = 0  # record total number of samples
    for i, batch in enumerate(dataloader):
        batch_size = batch[0].shape[0]
        batch_data = tuple(data.to(device) for data in batch)
        # Forward pass
        func_output = func(batch_data)
        loss, mask = (
            func_output,
            torch.ones(batch_size, 1),
        )

        # Here, we assume the loss is obtained with reduction="mean" by default.
        # We multiply the loss by batch_size to get the original gradient.
        loss *= batch_size
        # Backward pass
        loss.backward(retain_graph=True)

        with torch.no_grad():
            # Estimate covariance
            covariances = _update_covariance(
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
    covariances: Dict[str, Tuple[torch.Tensor]],
) -> Dict[str, Tuple[torch.Tensor]]:
    """Perform eigenvalue decomposition to covarince matrices.

    Args:
        covariances (Dict[str, Tuple[torch.Tensor]]): A dict that
            contains a pair of estimated covariance for each module.

    Returns:
        Dict[str, Tuple[torch.Tensor]]: A dict that contains a
            pair of eigenvector matrices for each module.
    """
    cached_q = {}
    for layer_name, (cov_a, cov_s) in covariances.items():
        _, q_a = torch.linalg.eigh(cov_a, UPLO="U")
        _, q_s = torch.linalg.eigh(cov_s, UPLO="U")
        cached_q[layer_name] = (q_a, q_s)

    return cached_q


def estimate_lambda(
    func: Callable,
    dataloader: torch.utils.data.DataLoader,
    eigenvectors: Dict[str, Tuple[torch.tensor]],
    layer_cache: Dict[str, Tuple[torch.tensor]],
    max_iter: Optional[int] = None,
    device: Optional[str] = "cpu",
) -> Dict[str, torch.tensor]:
    """Estimate the corrected eigenvalues in EK-FAC IFVP.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return the following,
            - loss: a single tensor of loss. Should be the mean loss by the
                    batch size.
            - mask (optional): a tensor of shape (batch_size, t), where 1's
                               indicate that the IFVP will be estimated on these
                               input positions and 0's indicate that these positions
                               are irrelevant (e.g. padding tokens).
            t is the number of steps, or sequence length of the input data. If the
            input data are non-sequential, t should be set to 1.
            The FIM will be estimated on this function.
        dataloader (torch.utils.data.DataLoader): The dataloader with full training
            samples for FIM estimation.
        eigenvectors (Dict[str, Tuple[torch.tensor]]): A dict that contains a
            pair of eigenvector matrices for each module.
        layer_cache (Dict[str, Tuple[torch.tensor]]): A dict that caches a pair
            of (inputs, outputs) for each module during the forward process.
        max_iter (Optional[int]): An integer indicating the maximum number of
            batches that will be used for estimating the lambdas.
        device (Optional[str]): Device to run the attributor on. Default is "cpu".

    Returns:
        Dict[str, torch.tensor]: A dict that contains the estimated lambda
            for each module.
    """
    if max_iter is None:
        max_iter = len(dataloader)

    lambdas = {}
    total_samples = 0  # record total number of samples
    for i, batch in enumerate(dataloader):
        batch_size = batch[0].shape[0]
        batch_data = tuple(data.to(device) for data in batch)
        # Forward pass
        func_output = func(batch_data)
        loss, mask = (
            func_output,
            torch.ones(batch_size, 1),
        )

        # Here, we assume the loss is obtained with reduction="mean" by default.
        # We multiply the loss by batch_size to get the original gradient.
        loss *= batch_size
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
        total_samples += batch_size
        if i == max_iter - 1:
            break

    return lambdas


class IFVPUsageError(Exception):
    """The usage exception class for IFVP module."""
