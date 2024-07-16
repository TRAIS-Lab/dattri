"""IFVP calculation functions.

IFVP (inverse FIM-vector product)
FIM (Fisher Information Matrix)

This module contains:
- `ifvp_explicit`: IFVP via explicit FIM calculation.
- `ifvp_at_x_explicit`: IFVP via explicit FIM calculation (with fixed x).
- `ifvp_datainf`: DataInf IFVP algorithm function.
- `ifvp_at_x_datainf`: DataInf IFVP algorithm function (with fixed x).
- `ifvp_at_x_ekfac`: EK-FAC IFVP algorithm function (with fixed x).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ClassVar, Generator, List, Optional, Tuple, Union

    from torch import Tensor


import warnings
from functools import wraps

import torch
from torch.func import grad, vmap


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
            calculate the emperial FIM, you should use the groundtruth label for
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
            calculate the emperial FIM, you should use the groundtruth label for
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


def ifvp_datainf(
    func: Callable,
    argnums: int,
    in_dims: Tuple[Union[None, int], ...],
    regularization: Optional[Union[float, List[float]]] = None,
    param_layer_map: Optional[List[int]] = None,
) -> Callable:
    """DataInf IFVP algorithm function.

    Standing for the inverse-FIM-vector product, returns a function that,
    when given vectors, computes the product of inverse-FIM and vector.

    DataInf assume the loss to be cross-entropy and thus derive a closed form
    IFVP without having to approximate the FIM. Implementation for reference:
    https://github.com/ykwon0407/DataInf/blob/main/src/influence.py

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The layer-wise gradients will
            be calculated on this function. Note that datainf expects the loss
            to be cross-entropy.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse FIM with respect to.
        in_dims (Tuple[Union[None, int], ...]): Parameter sent to vmap to produce
            batched layer-wise gradients. Example: inputs, weights, labels corresponds
            to (0,None,0).
        regularization (List [float]): A float or list of floats default to 0.0.
            Specifies the
            regularization term to be added to the Hessian matrix in each layer.
            This is useful when the Hessian matrix is singular or ill-conditioned.
            The regularization term is `regularization * I`, where `I` is the
            identity matrix directly added to the Hessian matrix. The list is
            of length L, where L is the total number of layers.
        param_layer_map: Optional[List[int]]: Specifies how the parameters are grouped
            into layers. Should be the same length as parameters tuple. For example,
            for a two layer model, params = (0.weights1,0.bias,1.weights,1.bias),
            param_layer_map should be [0,0,1,1],resulting in two layers as expected.

    Returns:
        A function that takes a list of tuples of Tensor `x` and a tuple of tensors
        `v`(layer-wise) and returns the approximated IFVP of the approximated Hessian of
        `func` and `v`.

    Raises:
        IFVPUsageError: If the length of regularization is not the same as the number
            of layers.
    """
    # TODO: param_layer_map should not be optional.

    batch_grad_func = vmap(grad(func, argnums=argnums), in_dims=in_dims)
    if regularization is not None and not isinstance(regularization, list):
        regularization = [regularization] * len(param_layer_map)

    if param_layer_map is not None and len(regularization) != len(param_layer_map):
        error_msg = "The length of regularization should\
                     be the same as the number of layers."
        raise IFVPUsageError(error_msg)

    def _single_datainf_ifvp(
        v: torch.Tensor,
        grad: torch.Tensor,
        regularization: float,
    ) -> torch.Tensor:
        # TODO: docstring
        coef = (v @ grad) / (regularization + torch.sum(grad**2))
        return (v - coef.reshape(-1, 1) @ grad.reshape(1, -1)) / regularization

    def _ifvp_datainf_func(
        x: Tuple[torch.Tensor, ...],
        v: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor]:
        """The IFVP function using DataInf.

        Args:
            x (Tuple[torch.Tensor, ...]): The function will compute the
                inverse FIM with respect to these arguments.
            v (Tuple[torch.Tensor, ...]): Tuple of layer-wise tensors from
                which IFVP will becomputed. For example layer-wise gradients
                of test samples.

        Returns:
            Layer-wise IFVP values.
        """
        grads = batch_grad_func(*x)
        layer_cnt = len(grads)
        if param_layer_map is not None:
            grouped = []
            max_layer = max(param_layer_map)
            for group in range(max_layer + 1):
                grouped_layers = tuple(
                    [
                        grads[layer]
                        for layer in range(len(param_layer_map))
                        if param_layer_map[layer] == group
                    ],
                )
                concated_grads = torch.concat(grouped_layers, dim=1)
                grouped.append(concated_grads)
            grads = tuple(grouped)
            layer_cnt = max_layer + 1  # Assuming count starts from 0
        ifvps = []
        for layer in range(layer_cnt):
            grad_layer = grads[layer]
            reg = 0.0 if regularization is None else regularization[layer]
            ifvp_contributions = vmap(
                lambda grad, layer=layer, reg=reg: _single_datainf_ifvp(
                    v[layer],
                    grad,
                    reg,
                ),
            )(grad_layer)
            ifvp_at_layer = ifvp_contributions.mean(dim=0)
            ifvps.append(ifvp_at_layer)
        return tuple(ifvps)

    return _ifvp_datainf_func


def ifvp_at_x_datainf(
    func: Callable,
    argnums: int,
    in_dims: Tuple[Union[None, int], ...],
    regularization: Optional[List[float]] = None,
    *x,
    param_layer_map: Optional[List[int]] = None,
) -> Callable:
    """DataInf IFVP algorithm function (with fixed x).

    Standing for the inverse-FIM-vector product, returns a function that,
    when given vectors, computes the product of inverse-FIM and vector.

    DataInf assume the loss to be cross-entropy and thus derive a closed form
    IFVP without having to approximate the FIM.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The layer-wise gradients will
            be calculated on this function. Note that datainf expects the loss
            to be cross-entropy.
        argnums (int): An integer default to 0. Specifies which argument of func
            to compute inverse FIM with respect to.
        in_dims (Tuple[Union[None, int], ...]): Parameter sent to vmap to produce
            batched layer-wise gradients. Example: inputs, weights, labels corresponds
            to (0,None,0).
        regularization (List [float]): A list of floats default to 0.0. Specifies the
            regularization term to be added to the Hessian matrix in each layer.
            This is useful when the Hessian matrix is singular or ill-conditioned.
            The regularization term is `regularization * I`, where `I` is the
            identity matrix directly added to the Hessian matrix.
            The list is of length L, where L is the total number of
            layers.
        param_layer_map: Optional[List[int]]: Specifies how the parameters are grouped
            into layers. Should be the same length as parameters tuple. For example,
            for a two layer model, params = (0.weights1,0.bias,1.weights,1.bias),
            param_layer_map should be (0,0,1,1),resulting in two layers as expected.
        *x: List of arguments for `func`.

    Returns:
        A function that takes a tuple `v` and returns the tuple of IFVPs of the Hessian
        of `func` and `v`.
    """

    def _per_sample_grad(*args) -> torch.Tensor:
        return grad(func, argnums=argnums)(*args)

    def _single_datainf_ifvp(
        v: torch.Tensor,
        grad: torch.Tensor,
        regularization: float,
    ) -> torch.Tensor:
        # TODO: same as the `_single_datainf_ifvp` defined in `ifvp_datainf`.
        coef = (v.T @ grad) / (regularization + torch.sum(grad**2))
        return (v - coef * grad) / regularization

    grads = vmap(_per_sample_grad, in_dims=in_dims)(*x)
    layer_cnt = len(grads)
    if param_layer_map is not None:
        grouped = []
        max_layer = max(param_layer_map)
        for group in range(max_layer + 1):
            grouped_layers = tuple(
                [
                    grads[layer]
                    for layer in range(len(param_layer_map))
                    if param_layer_map[layer] == group
                ],
            )
            concated_grads = torch.concat(grouped_layers, dim=1)
            grouped.append(concated_grads)
        grads = tuple(grouped)
        layer_cnt = max_layer + 1  # Assuming count starts from 0

    def _ifvp_at_x_datainf_func(
        v: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor]:
        """The IFVP function using datainf.

        Args:
            v (Tuple[torch.Tensor, ...]): Tuple of layer-wise tensors from
                which IFVP will becomputed. For example layer-wise gradients
                of test samples.

        Returns:
            The IFVP value dictionary, with keys corresponding to layer names.
        """
        # TODO: seems to be redundant if we have `_ifvp_datainf_func``
        # some code might be reused.
        ifvps = []
        for layer in range(layer_cnt):
            reg = 0.0 if regularization is None else regularization[layer]
            grad_layer = grads[layer]
            ifvp_contributions = vmap(
                lambda grad, layer=layer, reg=reg: _single_datainf_ifvp(
                    v[layer],
                    grad,
                    regularization=reg,
                ),
            )(grad_layer)
            ifvp_at_layer = ifvp_contributions.mean(dim=0)
            ifvps.append(ifvp_at_layer)
        return tuple(ifvps)

    return _ifvp_at_x_datainf_func


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


def _estimate_covariance(
    curr_estimate: List[List[Tuple[torch.Tensor]]],
    mlp_cache: List[MLPCache],
    total_samples: int,
    mask: torch.Tensor,
) -> List[List[Tuple[torch.Tensor]]]:
    """Estimate the 'covariance' matrices S and A in EK-FAC IFVP.

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
            if idx >= len(layer_cov):
                # First time initializartion
                layer_cov.append((batch_cov_a, batch_cov_s))
            else:
                old_weight = total_samples / (total_samples + batch_samples)
                new_weight = batch_samples / (total_samples + batch_samples)
                new_cov_a = old_weight * layer_cov[idx][0] + new_weight * batch_cov_a
                new_cov_s = old_weight * layer_cov[idx][1] + new_weight * batch_cov_s
                layer_cov[idx] = (new_cov_a, new_cov_s)

    return curr_estimate


def _estimate_lambda(
    curr_estimate: List[List[torch.Tensor]],
    mlp_cache: List[MLPCache],
    cached_q: List[List[Tuple[torch.Tensor]]],
    total_samples: int,
    mask: torch.Tensor,
    max_steps_for_vec: int = 10,
) -> List[List[torch.Tensor]]:
    """Estimate the corrected eigenvalues in EK-FAC IFVP.

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
    for cache, layer_lambda, layer_q in zip(mlp_cache, curr_estimate, cached_q):
        for idx, ((a_prev, s_curr), (q_a, q_s)) in enumerate(
            zip(cache.input_hidden_pairs, layer_q),
        ):
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

                for ts in range(timesteps):
                    batch_dtheta += (
                        ds_curr_reshaped[:, ts, :, None]
                        @ a_prev_reshaped[:, ts, None, :]
                    )

            # An equivalent way to calculate lambda's. Please refer to
            # https://math.stackexchange.com/questions/1879933/vector-multiplication-with-multiple-kronecker-products
            batch_lambda = torch.square(q_s @ batch_dtheta @ q_a.T).mean(axis=0)

            # Update the running eigenvalue estimation
            if idx >= len(layer_lambda):
                # First time initializartion
                layer_lambda.append(batch_lambda)
            else:
                old_weight = total_samples / (total_samples + batch_samples)
                new_weight = batch_samples / (total_samples + batch_samples)
                layer_lambda[idx] = (
                    old_weight * layer_lambda[idx] + new_weight * batch_lambda
                )

    return curr_estimate


def ifvp_at_x_ekfac(
    func: Callable,
    *x,
    in_dims: Optional[Tuple] = None,
    batch_size: int = 1,
    max_iter: Optional[int] = None,
    mlp_cache: Union[MLPCache, List[MLPCache]],
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

    cov_matrices = [[] for _ in range(len(mlp_cache))]
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
            cov_matrices = _estimate_covariance(
                cov_matrices,
                mlp_cache,
                total_samples,
                mask,
            )

        total_samples += int(mask.sum())
        if i == max_iter - 1:
            break

    # 2. Calculate the eigenvalue decomposition of S and A
    cached_q = [[] for _ in range(len(mlp_cache))]
    for layer_q, layer_cov in zip(cached_q, cov_matrices):
        for cov_a, cov_s in layer_cov:
            _, q_a = torch.linalg.eigh(cov_a, UPLO="U")
            _, q_s = torch.linalg.eigh(cov_s, UPLO="U")
            layer_q.append((q_a, q_s))

    # 3. Use random batch for eigenvalue correction
    dataloader = _random_batch_iterator(
        *x,
        num_samples=num_samples,
        in_dims=in_dims,
        batch_size=batch_size,
    )
    cached_lambdas = [[] for _ in range(len(mlp_cache))]
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
            cached_lambdas = _estimate_lambda(
                cached_lambdas,
                mlp_cache,
                cached_q,
                total_samples,
                mask,
            )
        total_samples += len(losses)
        if i == max_iter - 1:
            break

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
