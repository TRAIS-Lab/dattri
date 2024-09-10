"""This module contains functions that trigger model dropouts during test-time."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from typing import List

    import torch


# scenario 1: model already has dropout layers, then activate those during test-time
# just input the place we want to put, can support "all" and "specific"
def activate_dropout(
    model: torch.nn.Module,
    layer_positions: Optional[List[str]] = None,
    dropout_prob: float = 0.1,
) -> torch.nn.Module:
    """Activate dropout layers in the model.

    Args:
        model (torch.nn.Module): The model to be used.
        layer_positions (Optional[List[str]]): Specific layer positions to
            activate dropouts. Should be a list of layer names. Default to
            all dropout layers found in the model.
        dropout_prob (float): The dropout probability to be applied. If not
            specified, it will be defaulted as 0.1.

    Raises:
        ValueError: Some input layers do not contain nn.Dropout module.

    Returns:
        torch.nn.Module: The model with activated dropout layers.
    """
    # evaluate the model
    model.eval()
    # activate dropout layers
    if layer_positions is None:
        layer_positions = []

    invalid_layer_name = []
    if len(layer_positions) == 0:
        # activate all dropout layers found in the model
        for module in model.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.p = dropout_prob
                module.train()
    else:
        # activate all dropout layers in given positions
        for name, module in model.named_modules():
            for layer_name in layer_positions:
                # if the layer name (string) in name of modules,
                # then make dropout module in train mode
                if layer_name in name:
                    if module.__class__.__name__.startswith("Dropout"):
                        module.p = dropout_prob
                        module.train()
                    else:
                        invalid_layer_name.append(layer_name)

    if invalid_layer_name:
        invalid_layers_str = ", ".join(invalid_layer_name)
        msg = f"""Input layer names: {invalid_layers_str}
            do not correspond to dropout modules."""
        raise ValueError(msg)

    return model
