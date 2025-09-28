"""Common utility functions for gradient computation and influence attribution."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

# Configure logger
logger = logging.getLogger(__name__)


def stable_inverse(matrix: torch.Tensor, damping: Optional[float] = None) -> torch.Tensor:
    """Compute a numerically stable inverse of a matrix using eigendecomposition.

    Args:
        matrix: Input matrix to invert
        damping: (Adaptive) Damping factor for numerical stability

    Returns:
        Stable inverse of the input matrix with the same dtype as input
    """
    orig_dtype = matrix.dtype
    matrix = matrix.to(dtype=torch.float32)

    assert matrix.dim() == 2, "Input must be a 2D matrix"

    # Add damping to the diagonal
    if damping is None:
        damping = 1e-5 * torch.trace(matrix) / matrix.size(0)
    else:
        damping = damping * torch.trace(matrix) / matrix.size(0)

    damped_matrix = matrix + damping * torch.eye(matrix.size(0), device=matrix.device)

    try:
        L = torch.linalg.cholesky(damped_matrix)
        inverse = torch.cholesky_inverse(L)
    except RuntimeError:
        logger.warning("Falling back to direct inverse due to Cholesky failure")
        inverse = torch.inverse(damped_matrix)

    return inverse.to(dtype=orig_dtype)


def vectorize(
    g: Dict[str, torch.Tensor],
    batch_dim: Optional[bool] = True,
    arr: Optional[torch.Tensor] = None,
    device: Optional[str] = "cuda",
) -> torch.Tensor:
    """Vectorize gradients into a flattened tensor.

    This function takes a dictionary of gradients and returns a flattened tensor
    of shape [batch_size, num_params].

    Args:
        g: A dictionary containing gradient tensors to be vectorized
        batch_dim: Whether to include the batch dimension in the returned tensor
        arr: An optional pre-allocated tensor to store the vectorized gradients
        device: The device to store the tensor on

    Returns:
        torch.Tensor: A flattened tensor of gradients

    Raises:
        ValueError: If parameter row num doesn't match batch size when batch_dim=True.
    """
    if arr is None:
        if batch_dim:
            g_elt = g[next(iter(g.keys()))]
            batch_size = g_elt.shape[0]
            num_params = 0
            for param in g.values():
                if param.shape[0] != batch_size:
                    msg = "Parameter row num doesn't match batch size."
                    raise ValueError(msg)
                num_params += int(param.numel() / batch_size)
            arr = torch.empty(
                size=(batch_size, num_params),
                dtype=g_elt.dtype,
                device=device,
            )
        else:
            num_params = 0
            for param in g.values():
                num_params += int(param.numel())
            arr = torch.empty(size=(num_params,), dtype=param.dtype, device=device)

    pointer = 0
    vector_dim = 1
    for param in g.values():
        if batch_dim:
            if len(param.shape) <= vector_dim:
                num_param = 1
                p = param.data.reshape(-1, 1)
            else:
                num_param = param[0].numel()
                p = param.flatten(start_dim=1).data
            arr[:, pointer : pointer + num_param] = p.to(device)
            pointer += num_param
        else:
            num_param = param.numel()
            arr[pointer : pointer + num_param] = param.reshape(-1).to(device)
            pointer += num_param
    return arr


def get_parameter_chunk_sizes(
    param_shape_list: List,
    batch_size: int,
) -> tuple[int, List[int]]:
    """Compute chunk size information from feature to be projected.

    Get a tuple containing max chunk size and a list of the number of
    parameters in each chunk.

    Args:
        param_shape_list: A list of numbers indicating the total number of
            features to be projected. A typical example is a list of parameter
            size of each module in a torch.nn.Module model.
        batch_size: The batch size. Each term (or module) in feature
            will have the same batch size.

    Returns:
        tuple[int, List[int]]: A tuple containing:
            - Maximum number of parameters per chunk
            - A list of the number of parameters in each chunk
    """
    # get the number of total params
    param_num = param_shape_list[0]

    max_chunk_size = np.iinfo(np.uint32).max // batch_size

    num_chunk = param_num // max_chunk_size
    remaining = param_num % max_chunk_size
    params_per_chunk = (
        [max_chunk_size] * num_chunk + [remaining]
        if remaining > 0
        else [max_chunk_size] * num_chunk
    )

    return max_chunk_size, params_per_chunk


def find_layers(model, layer_type="Linear", return_type="instance"):
    """Find layers of specified type in a model.

    Args:
        model: PyTorch model to search
        layer_type: Type of layer to find ('Linear', 'LayerNorm', or 'Linear_LayerNorm')
        return_type: What to return ('instance', 'name', or 'name_instance')

    Returns:
        List of layers, layer names, or (name, layer) tuples

    Raises:
        ValueError: If layer_type is not one of 'Linear', 'LayerNorm', or 'Linear_LayerNorm'.
    """
    layers = []
    return_module_name = return_type != "instance"

    if return_module_name:
        for module_name, module in model.named_modules():
            if isinstance(
                module,
                (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.Linear),
            ):
                layers.append((module_name, module))
    else:
        layers.extend(module for module in model.modules() if isinstance(
                module,
                (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.Linear),
            ))

    if return_module_name:
        if layer_type == "Linear":
            layers = [
                (name, layer)
                for name, layer in layers
                if isinstance(layer, torch.nn.Linear)
            ]
        elif layer_type == "Linear_LayerNorm":
            layers = [
                (name, layer)
                for name, layer in layers
                if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm))
            ]
        elif layer_type == "LayerNorm":
            layers = [
                (name, layer)
                for name, layer in layers
                if isinstance(layer, torch.nn.LayerNorm)
            ]
        else:
            msg = "Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'."
            raise ValueError(
                msg,
            )
    elif layer_type == "Linear":
        layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
    elif layer_type == "Linear_LayerNorm":
        layers = [
            layer
            for layer in layers
            if isinstance(layer, (torch.nn.LayerNorm, torch.nn.Linear))
        ]
    elif layer_type == "LayerNorm":
        layers = [layer for layer in layers if isinstance(layer, torch.nn.LayerNorm)]
    else:
        msg = "Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'."
        raise ValueError(
            msg,
        )

    if return_type == "instance":
        return layers
    if return_type == "name":
        return [name for name, layer in layers]
    if return_type == "name_instance":
        return [(name, layer) for name, layer in layers]
    msg = "Invalid return_type. Choose from 'instance', 'name', and 'name_instance'."
    raise ValueError(
        msg,
    )
