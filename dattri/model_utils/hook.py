"""This module contains functions that add/remove hooks on pyTorch models."""
# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple

import torch
from torch import Tensor


def get_layer_feature(
    model: torch.nn.Module,
    layer_name: str,
    dataloader: torch.utils.data.DataLoader,
) -> Tensor:
    """Get the feature at layer_name for data passing through a model.

    Args:
        model (torch.nn.Module): The model used for forward-passing.
        layer_name (str): The location that we want to extract the feauture.
            A typical usage is to extract the last intermediate layer feature,
            which corresponds to the "final feature" used for prediction.
        dataloader (torch.utils.data.DataLoader): The dataloader that contains
            the data points in interest to obtain the feature at layer_name.

    Returns:
        A Tensor containing the output feature of the data in the provided
            dataloaderat the specific layer of the model.
    """
    layer_feature = None

    def _forward_hook(_model: torch.nn.Module, _input: Tuple, output: Tensor) -> None:
        """Forward hook function to extract layer features.

        Args:
            _model (torch.nn.Module): The PyTorch model in interest. Will not
                be used in this function.
            _input (Tuple): Input tensors to the module. Will not be used in
                this function.
            output (Tensor): The output tensor of the module.
        """
        nonlocal layer_feature
        layer_feature = output.detach()

    model_layer = getattr(model, layer_name)

    hook_handle = model_layer.register_forward_hook(_forward_hook)

    feature_list = []
    for x, _ in dataloader:
        _ = model(x)
        feature_list.append(layer_feature)

    hook_handle.remove()
    return torch.cat(feature_list, dim=0)
