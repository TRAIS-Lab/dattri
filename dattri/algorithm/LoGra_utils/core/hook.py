"""
Hook manager for efficient gradient component capture and projection.
Simplified version with sparsification removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import functools
import time
import logging

import torch.jit as jit

# Configure logger
logger = logging.getLogger(__name__)

@jit.script
def compute_linear_gradients_2d(grad_pre_activation: Tensor, input_features: Tensor) -> Tensor:
    """
    Compute weight gradients using outer products for 2D tensors.

    Args:
        grad_pre_activation: Gradient of pre-activation with shape [batch_size, output_dim]
        input_features: Input features with shape [batch_size, input_dim]

    Returns:
        Tensor of shape [batch_size, output_dim * input_dim] containing per-sample gradients
    """
    batch_size = input_features.shape[0]
    output_dim = grad_pre_activation.shape[1]
    input_dim = input_features.shape[1]

    grad_tensor = torch.einsum('bi,bj->bij', grad_pre_activation, input_features)
    return grad_tensor.reshape(batch_size, output_dim * input_dim)

@jit.script
def compute_linear_gradients_3d(grad_pre_activation: Tensor, input_features: Tensor) -> Tensor:
    """
    Compute weight gradients using outer products for 3D tensors (sequence data).

    Args:
        grad_pre_activation: Gradient of pre-activation with shape [batch_size, seq_length, output_dim]
        input_features: Input features with shape [batch_size, seq_length, input_dim]

    Returns:
        Tensor of shape [batch_size, output_dim * input_dim] containing per-sample gradients
    """
    batch_size = input_features.shape[0]
    output_dim = grad_pre_activation.shape[2]
    input_dim = input_features.shape[2]

    grad_tensor = torch.einsum('bsi,bsj->bij', grad_pre_activation, input_features)
    return grad_tensor.reshape(batch_size, output_dim * input_dim)

class HookManager:
    """
    Manages hooks for efficient gradient component capturing and projection.
    Simplified version without sparsification functionality.
    """
    def __init__(
            self,
            model: nn.Module,
            layer_names: List[str],
            profile: bool = False,
            device: str = 'cpu'
        ) -> None:
        """
        Initialize the hook manager

        Args:
            model: The model to hook
            layer_names: Names of layers to hook
            profile: Whether to profile execution time
            device: Device to use for profiling synchronization
        """
        self.model = model
        self.layer_names = layer_names
        self.profile = profile
        self.device = device

        # Create mapping from layer name to index for O(1) lookups
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(layer_names)}

        self.forward_hooks = [None] * len(layer_names)
        self.backward_hooks = [None] * len(layer_names)
        self.compressed_grads = [None] * len(layer_names)
        self.inputs = [None] * len(layer_names)
        self.pre_activations = [None] * len(layer_names)
        self.normalized = [None] * len(layer_names)
        self.projectors = [None] * len(layer_names)

        # Profiling stats
        self.compression_time = 0.0

        # Register hooks
        self._register_hooks()

        logger.info(f"Initialized HookManager with {len(layer_names)} layer hooks")

    def _register_hooks(self):
        """Register forward and backward hooks to target layers"""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]

                # Use functools.partial to correctly bind parameters to avoid late binding issues
                forward_hook = functools.partial(self._forward_hook_fn, name)
                backward_hook = functools.partial(self._backward_hook_fn, name)

                # Register hooks with properly bound parameters
                self.forward_hooks[idx] = module.register_forward_hook(forward_hook)
                self.backward_hooks[idx] = module.register_full_backward_hook(backward_hook)

                logger.debug(f"Registered hooks for layer: {name}")

    def set_projectors(self, projectors: List[Any]) -> None:
        """
        Set projector objects for each layer

        Args:
            projectors: List of projector objects, ordered by layer_names
        """
        self.projectors = projectors
        logger.debug(f"Set {len(projectors)} projectors for HookManager")

    def get_compressed_grads(self) -> List[Tensor]:
        """
        Get all captured projected gradients

        Returns:
            List of projected gradient tensors, ordered by layer_names
        """
        return self.compressed_grads

    def get_compression_time(self) -> float:
        """
        Get the accumulated projection time

        Returns:
            Total time spent in projection operations
        """
        return self.compression_time

    def _forward_hook_fn(self, name: str, mod: nn.Module, inp: Any, out: Any) -> None:
        """
        Forward hook function that captures inputs and pre-activations

        Args:
            name: Layer name
            mod: Module instance
            inp: Input tensors
            out: Output tensors
        """
        # Get the index for this layer
        idx = self.layer_name_to_idx[name]

        # Store input
        if isinstance(inp, tuple) and len(inp) > 0:
            self.inputs[idx] = inp[0].detach()
        else:
            self.inputs[idx] = inp.detach()

        # Store pre-activation (output)
        self.pre_activations[idx] = out.detach()

        # For LayerNorm, also capture the normalized tensor
        if isinstance(mod, nn.LayerNorm):
            x = inp[0] if isinstance(inp, tuple) else inp
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            normalized = (x - mean) / torch.sqrt(var + mod.eps)
            self.normalized[idx] = normalized.detach()

    def _backward_hook_fn(self, name: str, mod: nn.Module, grad_input: Any, grad_output: Any) -> None:
        """
        Backward hook function that computes projected gradients

        Args:
            name: Layer name
            mod: Module instance
            grad_input: Gradient w.r.t inputs
            grad_output: Gradient w.r.t outputs
        """
        # Get the index for this layer
        idx = self.layer_name_to_idx[name]

        # Get the gradient of the pre-activation
        grad_pre_activation = grad_output[0]

        # Calculate the projected gradient based on layer type
        with torch.no_grad():
            if isinstance(mod, nn.Linear):
                grad = self._linear_grad(
                    mod, idx, grad_pre_activation, per_sample=True
                )
            elif isinstance(mod, nn.LayerNorm):
                grad = self._layernorm_grad(
                    mod, idx, grad_pre_activation, per_sample=True
                )
            elif isinstance(mod, nn.Embedding):
                # Embeddings would need their own implementation
                grad = None
            else:
                # Fallback for other layer types
                grad = None

            if grad is not None:
                # Store the projected gradient
                self.compressed_grads[idx] = grad.detach()

    def _linear_grad(
        self,
        layer: nn.Linear,
        idx: int,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute the gradient for Linear layers with projection.
        Simplified version without sparsification.

        Args:
            layer: Linear layer
            idx: Layer index
            grad_pre_activation: Gradient of the pre-activation
            per_sample: Whether to compute per-sample gradients

        Returns:
            Projected gradient tensor
        """
        input_features = self.inputs[idx]
        is_3d = input_features.dim() == 3

        # Get projector for this layer
        projector = self.projectors[idx] if hasattr(self, 'projectors') and idx < len(self.projectors) else None

        # Process tensors for gradient computation
        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            # Reshape 3D tensors to 2D for consistent processing
            input_features_flat = input_features.reshape(-1, hidden_size)
            grad_pre_activation_flat = grad_pre_activation.reshape(-1, layer.out_features)
        else:
            batch_size = input_features.shape[0]
            input_features_flat = input_features
            grad_pre_activation_flat = grad_pre_activation

        # Scale the gradient if we're computing per-sample gradients
        if per_sample:
            grad_pre_activation_flat = grad_pre_activation_flat * batch_size

        # Handle bias term by augmenting input with ones
        if layer.bias is not None:
            ones = torch.ones(
                input_features_flat.size(0), 1,
                device=input_features_flat.device,
                dtype=input_features_flat.dtype
            )
            input_features_flat = torch.cat([input_features_flat, ones], dim=1)

        # Reshape back to 3D if needed
        if is_3d:
            input_features_3d = input_features_flat.reshape(batch_size, seq_length, -1)
            grad_pre_activation_3d = grad_pre_activation_flat.reshape(batch_size, seq_length, -1)

        # Start timing for projection if profiling is enabled
        if self.profile:
            torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
            start_time = time.time()

        # Compute the outer product to get the gradient
        if is_3d:
            grad_tensor = compute_linear_gradients_3d(grad_pre_activation_3d, input_features_3d)
        else:
            grad_tensor = compute_linear_gradients_2d(grad_pre_activation_flat, input_features_flat)

        grad = grad_tensor.reshape(batch_size, -1)

        # Apply projector if available
        if projector is not None:
            grad = projector.projector_grad(grad)

        # End timing for projection
        if self.profile:
            torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
            self.compression_time += time.time() - start_time

        return grad

    def _layernorm_grad(
        self,
        layer: nn.LayerNorm,
        idx: int,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute the gradient for LayerNorm layers

        Args:
            layer: LayerNorm layer
            idx: Layer index
            grad_pre_activation: Gradient of the pre-activation
            per_sample: Whether to compute per-sample gradients

        Returns:
            Projected gradient tensor
        """
        if not layer.elementwise_affine:
            return None

        normalized = self.normalized[idx]
        if normalized is None:
            return None

        is_3d = normalized.dim() == 3

        if per_sample:
            grad_pre_activation = grad_pre_activation * normalized.shape[0]
            if is_3d:
                grad_weight = torch.einsum("ijk,ijk->ik", grad_pre_activation, normalized)
                grad_bias = torch.sum(grad_pre_activation, dim=1)
            else:
                grad_weight = grad_pre_activation * normalized
                grad_bias = grad_pre_activation
        else:
            if is_3d:
                grad_weight = torch.sum(grad_pre_activation * normalized, dim=(0, 1))
                grad_bias = torch.sum(grad_pre_activation, dim=(0, 1))
            else:
                grad_weight = torch.sum(grad_pre_activation * normalized, dim=0)
                grad_bias = torch.sum(grad_pre_activation, dim=0)

        # Concatenate weight and bias gradients
        grad = torch.cat((grad_weight, grad_bias), dim=1)

        # Apply projector if available
        projector = self.projectors[idx] if hasattr(self, 'projectors') and idx < len(self.projectors) else None
        if projector is not None and hasattr(projector, 'projector_grad') and projector.projector_grad is not None:
            # Start timing for projection if profiling is enabled
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                start_time = time.time()

            grad = projector.projector_grad(grad)

            # End timing for projection
            if self.profile:
                torch.cuda.synchronize(self.device) if torch.cuda.is_available() and self.device != 'cpu' else None
                self.compression_time += time.time() - start_time

        return grad

    def remove_hooks(self) -> None:
        """Remove all hooks"""
        for hook in self.forward_hooks:
            if hook is not None:
                hook.remove()
        for hook in self.backward_hooks:
            if hook is not None:
                hook.remove()
        self.forward_hooks = [None] * len(self.layer_names)
        self.backward_hooks = [None] * len(self.layer_names)
        logger.debug("Removed all hooks from HookManager")