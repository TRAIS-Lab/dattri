"""This module contains functions that add/remove hooks on PyTorch models."""
# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Tuple

    from torch.utils.data import DataLoader

import torch
from torch import Tensor


def _get_layer_feature(
    model: torch.nn.Module,
    layer_name: str,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor]:
    """Get the feature at layer_name for data passing through a model.

    Args:
        model (torch.nn.Module): The model used for forward-passing.
        layer_name (str): The location that we want to extract the feauture.
            A typical usage is to extract the last intermediate layer feature,
            which corresponds to the "final feature" used for prediction.
        dataloader (DataLoader): The dataloader that contains
            the data points in interest to obtain the feature at layer_name.
        device (str): The device for the output.

    Returns:
        A tuple containing (1)the output feature of the data at the specific layer
            of the model and (2)the model output.
    """
    feature_list: List[Tensor] = []
    output_list: List[Tensor] = []

    def _forward_hook(_model: torch.nn.Module, _input: Tuple, output: Tensor) -> None:
        """Forward hook function to extract layer features.

        Args:
            _model (torch.nn.Module): The PyTorch model in interest. Will not
                be used in this function.
            _input (Tuple): Input tensors to the module. Will not be used in
                this function.
            output (Tensor): The output tensor of the module.
        """
        feature_list.append(output.detach())

    model_layer = getattr(model, layer_name)
    hook_handle = model_layer.register_forward_hook(_forward_hook)

    with torch.no_grad():
        for x, _ in dataloader:
            out = model(x)
            output_list.append(out.detach())

    hook_handle.remove()
    return torch.cat(feature_list, dim=0).to(device), torch.cat(output_list, dim=0).to(
        device,
    )


def get_final_linear_layer_input(
    model: torch.nn.Module,
    final_linear_layer_name: str,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor]:
    """Get input and output of the final layer for data passing through a model.

    Args:
        model (torch.nn.Module): The model used for forward-passing.
        final_linear_layer_name (str): The layer name of the final linear layer.
            The input of this layer should be the output of the "prediction model"
            mentioned in the RPS paper.
        dataloader (DataLoader): The dataloader that contains
            the data points in interest.
        device (str): The device for the output.

    Returns:
        A tuple containing (1)the output feature of the prediction model and
            (2)the overall model output.
    """
    feature_list: List[Tensor] = []
    output_list: List[Tensor] = []

    def _forward_hook(_model: torch.nn.Module, _input: Tuple, output: Tensor) -> None:
        """Forward hook function to extract layer features.

        Args:
            _model (torch.nn.Module): The PyTorch model in interest. Will not
                be used in this function.
            _input (Tuple): Input tensors to the module.
            output (Tensor): The output tensor of the module.
        """
        feature_list.append(_input[0].detach())
        output_list.append(output.detach())

    model_layer = getattr(model, final_linear_layer_name)
    hook_handle = model_layer.register_forward_hook(_forward_hook)

    # perform a forward pass through the dataloader
    with torch.no_grad():
        for x, _ in dataloader:
            _ = model(x)

    hook_handle.remove()
    return torch.cat(feature_list, dim=0).to(device), torch.cat(output_list, dim=0).to(
        device,
    )
