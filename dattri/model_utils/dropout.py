"""This module contains functions that trigger model dropouts during test-time."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional

    import torch


# scenario 1: model already has dropout layers, then activate those during test-time
# just input the place we want to put, can support "all" and "specific"
def activate_dropout(
    model: torch.nn.Module,
    mode: str = "all",
    layer_positions: Optional[List[str]] = None,
) -> None:
    """Active dropout layers in the model.

    Args:
        model (torch.nn.Module): The model to be used.
        mode (str, optional): Where to active dropouts. Can be "all" or "specific".
            Defaults to "all".
        layer_positions (Optional[List[str]], optional): Specific layer positions to
            activate dropouts. Should be a list of layer names. Defaults to None.
    """
    if mode == "all":
        for module in model.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.train()

    elif mode == "specific":
        for name, module in model.named_modules():
            for layer_name in layer_positions:
                # if the layer name (string) in name of modules,
                # then make dropout module in train mode
                if layer_name in name and module.__class__.__name__.startswith(
                    "Dropout",
                ):
                    module.train()
