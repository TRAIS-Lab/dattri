"""Utility functions for working with models and parameters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Dict

import functools

import torch


def flatten_params(tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dictionary of tensors into a single tensor.

    This is useful for transforming model.named_parameters()
    results into a single tensor which can be passed to hvp or ihvp functions.

    Args:
        tensors: A dictionary of tensors. E.g., the result of model.named_parameters().

    Returns:
        (torch.Tensor): A single tensor containing the flattened parameters.

    Note:
        This function will flatten the tensors in the order they are passed in the
        dictionary and the flattened tensor will be a concatenation of all the tensors
        on one dimension.
    """
    return torch.cat(
        [t.reshape(-1) for t in tensors.values()],
        dim=-1,
    )


def _unflatten_params(tensors: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Unflatten a single tensor into a dictionary of tensors.

    This is a reverse operation of flatten_params. The transforming could enable the
    following usage of `functional_call` function.

    Args:
        tensors: A single tensor containing the flattened parameters.
        model: A torch.nn.Module object, providing the shape information and the names
            of the parameters.

    Returns:
        (dict[str, torch.Tensor]): A dictionary of tensors.
            E.g., the result of model.named_parameters().

    Note:
        The returned value will use the `tensor` as the value of the dictionary, rather
        than directly returning model.named_parameters().
    """
    model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    shape_list = [p.shape for p in model_params.values()]
    def generator() -> torch.Tensor:
        current_index = 0
        for shape in shape_list:
            size = math.prod(shape)
            yield tensors[current_index:current_index + size].reshape(shape)
            current_index += size
    return dict(zip(model_params.keys(), generator()))


def flatten_func(model: torch.nn.Module, param_num: int = 0) -> Callable:
    """A decorator that flattens the parameters of a function at a specified index.

    This is useful for working with functions that taking the dictionary of parameters
    as input, but we want to pass the flattened parameters to the function.

    Args:
        model: A torch.nn.Module object, providing the shape information and the names
            of the parameters.
        param_num: The index of the parameter that should be flattened.

    Returns:
        (Callable): A decorator that flattens the parameters of a function at the
            specified index.
    """
    def flatten_params_wrapper(function: Callable) -> Callable:
        """A wrapper function that flattens the parameters of a function.

        Args:
            function: The function to be wrapped.

        returns:
            (Callable): A wrapped function that flattens the parameters at the
                specified index.
        """
        @functools.wraps(function)
        def _function_flattened(*args, **kwargs: Dict[str, Any]) -> torch.Tensor:
            new_args = list(args)
            new_args[param_num] = _unflatten_params(args[param_num], model)
            return function(*new_args, **kwargs)
        return _function_flattened
    return flatten_params_wrapper
