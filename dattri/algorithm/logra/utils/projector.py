"""Projector container classes for gradient compression."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from dattri.func.projection import random_project

# Configure logger
logger = logging.getLogger(__name__)


class ProjectorContainer:
    """Container for projector functions associated with a layer.
    Used to store projectors without modifying the original layer.
    """

    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index
        self.projector_grad = None
        self.projector_grad_comp = (None, None)


def setup_model_projectors(
    model: nn.Module,
    layer_names: List[str],
    projector_kwargs: Dict[str, Any],
    train_dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> List[ProjectorContainer]:
    """Sets up projectors for each layer in the model.

    Args:
        model: The PyTorch model
        layer_names: Names of layers to set projectors for
        projector_kwargs: Keyword arguments for projector configuration
        train_dataloader: DataLoader for training data (used to get input shapes)
        device: Device to run the model on

    Returns:
        List of projector containers, ordered by layer_names

    Raises:
        ValueError: If an unsupported layer type is encountered.
    """
    if not projector_kwargs:
        return []

    # Extract configuration parameters
    proj_seed = projector_kwargs.get("proj_seed", 0)
    proj_factorize = projector_kwargs.get("proj_factorize", True)

    # Remove parameters that are handled separately
    kwargs_copy = projector_kwargs.copy()
    if "proj_seed" in kwargs_copy:
        kwargs_copy.pop("proj_seed")
    if 'proj_factorize' in kwargs_copy:
        kwargs_copy.pop("proj_factorize")

    # Initialize containers list
    projectors = [None] * len(layer_names)

    # Create name to index mapping for faster lookup
    name_to_index = {name: idx for idx, name in enumerate(layer_names)}

    # Run a forward pass to initialize model
    logger.info("Running forward pass to initialize model for projector setup")
    train_batch = next(iter(train_dataloader))
    if isinstance(train_batch, dict):
        inputs = {k: v.to(device) for k, v in train_batch.items()}
        model(**inputs)
    else:
        inputs = train_batch[0].to(device)
        model(inputs)

    # First, capture inputs and outputs for each layer
    layer_inputs = {}
    layer_outputs = {}
    hooks = []

    def capture_hook(name, mod, inp, out) -> None:
        layer_inputs[name] = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
        layer_outputs[name] = out

    # Register temporary hooks to capture layer I/O
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(
                lambda mod, inp, out, n=name: capture_hook(n, mod, inp, out),
            )
            hooks.append(hook)

    # Run another forward pass to capture inputs/outputs
    if isinstance(train_batch, dict):
        model(**inputs)
    else:
        model(inputs)

    # Remove temporary hooks
    for hook in hooks:
        hook.remove()

    # Create projectors for each layer
    for module_id, (module_name, module) in enumerate(model.named_modules()):
        if module_name in layer_names:
            idx = name_to_index[module_name]
            projector = ProjectorContainer(module_name, idx)
            base_seed = proj_seed + int(1e4) * module_id

            proj_kwargs = kwargs_copy.copy()
            # proj_kwargs["active_indices"] = None

            # Create appropriate projectors based on layer type
            if isinstance(module, nn.Linear):
                _setup_linear_projector(
                    projector,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs,
                    proj_factorize,
                )
            elif isinstance(module, nn.LayerNorm):
                _setup_layernorm_projector(
                    projector,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs,
                    proj_factorize,
                )
            else:
                msg = f"Unsupported layer type: {type(module)}"
                raise ValueError(msg)

            projectors[idx] = projector

    logger.info(f"Set up projectors for {len(layer_names)} layers")
    return projectors


def _setup_linear_projector(
    projector: ProjectorContainer,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    projector_kwargs: Dict[str, Any],
    proj_factorize: bool = True
) -> None:
    """Set up projector for a Linear layer.

    Args:
        projector: ProjectorContainer to store the projector
        layer: Linear layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        projector_kwargs: Keyword arguments for the projection
        proj_factorize: Whether to factorize the projection
    """
    if pre_activation is None or layer_input is None:
        return

    batch_size = pre_activation.shape[0]
    is_3d = layer_input.dim() == 3

    input_features = layer_input
    if layer.bias is not None:
        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            input_features = input_features.reshape(-1, hidden_size)
        else:
            batch_size = input_features.shape[0]

        ones = torch.ones(
            input_features.size(0),
            1,
            device=input_features.device,
            dtype=input_features.dtype,
        )
        input_features = torch.cat([input_features, ones], dim=1)

        if is_3d:
            input_features = input_features.reshape(batch_size, seq_length, -1)

    if proj_factorize:
        dumb_grad_comp_1 = torch.zeros_like(pre_activation.view(-1, pre_activation.shape[-1]))

        projector_grad_comp_1 = random_project(
            dumb_grad_comp_1,
            dumb_grad_comp_1.shape[0],
            proj_seed=base_seed,
            **{k: v for k, v in projector_kwargs.items() if k != 'active_indices'}
        )

        dumb_grad_comp_2 = torch.zeros_like(input_features.view(-1, input_features.shape[-1]))
        projector_grad_comp_2 = random_project(
            dumb_grad_comp_2,
            dumb_grad_comp_2.shape[0],
            proj_seed=base_seed + 1,
            **{k: v for k, v in projector_kwargs.items() if k != 'active_indices'}
        )

        projector.projector_grad_comp = (
            projector_grad_comp_1, projector_grad_comp_2
        )
    else:
        # Compute the outer product to get the gradient shape
        if is_3d:
            dumb_grad = torch.einsum(
                "ijk,ijl->ikl",
                pre_activation,
                input_features,
            ).reshape(batch_size, -1)
        else:
            dumb_grad = torch.einsum("bi,bj->bij", pre_activation, input_features).reshape(
                batch_size,
                -1,
            )

        # Create projector using original random_project function
        projector_grad = random_project(
            dumb_grad,
            dumb_grad.shape[0],
            proj_seed=base_seed,
            # pre_compute=proj_factorize,
            **projector_kwargs,
        )

        projector.projector_grad = projector_grad


def _setup_layernorm_projector(
    projector: ProjectorContainer,
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    projector_kwargs: Dict[str, Any],
    proj_factorize: bool = True
) -> None:
    """Set up projector for a LayerNorm layer.

    Args:
        projector: ProjectorContainer to store the projector
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        projector_kwargs: Keyword arguments for the projection
        proj_factorize: Whether to factorize the projection to 2 projection matrices
    """
    if not layer.elementwise_affine:
        return

    if pre_activation is None:
        return

    # For LayerNorm, gradient has shape (batch_size, 2 * normalized_shape)
    # because it includes both weight and bias gradients
    if proj_factorize:
        dumb_grad_comp_1 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))
        projector_grad_comp_1 = random_project(
            dumb_grad_comp_1,
            dumb_grad_comp_1.shape[0],
            proj_seed=base_seed,
            **projector_kwargs
        )

        dumb_grad_comp_2 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))
        projector_grad_comp_2 = random_project(
            dumb_grad_comp_2,
            dumb_grad_comp_2.shape[0],
            proj_seed=base_seed + 1,
            **projector_kwargs
        )

        projector.projector_grad_comp = (
            projector_grad_comp_1, projector_grad_comp_2
        )
    else:
        dumb_grad_comp = torch.zeros(
            (pre_activation.shape[0], pre_activation.shape[-1] * 2),
        )

        projector_grad = random_project(
            dumb_grad_comp,
            dumb_grad_comp.shape[0],
            proj_seed=base_seed,
            # pre_compute=proj_factorize,
            **projector_kwargs,
        )

        projector.projector_grad = projector_grad
