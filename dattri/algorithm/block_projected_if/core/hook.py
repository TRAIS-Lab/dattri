"""Hook manager for efficient gradient compression with prevented materialization.

This implementation uses monkey-patching with custom autograd Functions to prevent
full gradient materialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

    from .compressor import Compressor

import functools
import logging

import torch
import torch.nn.functional as F  # noqa: N812 - F is standard PyTorch convention
from torch import Tensor, nn
from torch.autograd import Function

logger = logging.getLogger(__name__)

# Global registry: maps a unique ID to hook manager
# Used to avoid storing hook_manager in autograd context, which would cause memory leaks
_HOOK_MANAGER_REGISTRY = {}


class CompressedLinearBackward(Function):
    """Custom autograd Function that prevents full gradient materialization.

    Key mechanism: When backward() returns None for a parameter, PyTorch skips
    computing that parameter's gradient entirely. This is how we avoid materializing
    the full weight gradient.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor_input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        hook_manager_id: int,
        layer_idx: int,
    ) -> Tensor:
        """Forward pass: standard linear transformation.

        We store hook_manager_id (an int) instead of the hook_manager object.
        Storing the object would keep it in the autograd graph, causing memory leaks.

        Args:
            ctx: Autograd context for saving tensors
            tensor_input: Input tensor to the linear layer
            weight: Weight tensor
            bias: Optional bias tensor
            hook_manager_id: ID for looking up hook manager in registry
            layer_idx: Index of the layer

        Returns:
            Output tensor from linear transformation

        Raises:
            RuntimeError: If hook manager not found in registry
        """
        ctx.save_for_backward(weight, bias)
        ctx.hook_manager_id = hook_manager_id
        ctx.layer_idx = layer_idx

        # Lookup hook manager from global registry
        hook_manager = _HOOK_MANAGER_REGISTRY.get(hook_manager_id)
        if hook_manager is None:
            msg = f"Hook manager {hook_manager_id} not found in registry"
            raise RuntimeError(msg)

        # Store input for backward pass (centralized storage)
        # Store at original dtype - will be cast during backward if needed
        hook_manager.inputs[layer_idx] = tensor_input.detach()

        # Cast input to weight dtype for computation (handles mixed precision)
        # This mimics PyTorch's autocast behavior
        input_compute = (
            tensor_input.to(weight.dtype)
            if tensor_input.dtype != weight.dtype
            else tensor_input
        )

        # Standard forward pass (same as nn.Linear)
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None, None, None]:
        """Backward pass: compute ONLY compressed gradient, not the full gradient.

        By returning None for weight and bias, we tell PyTorch: "don't compute
        param.grad for these, I handled it myself". This is the key mechanism
        that prevents full gradient materialization.

        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Tuple of gradients: (grad_input, None, None, None, None)

        Raises:
            RuntimeError: If hook manager not found in registry
        """
        weight, bias = ctx.saved_tensors
        hook_manager_id = ctx.hook_manager_id
        layer_idx = ctx.layer_idx

        # Lookup hook manager from registry
        hook_manager = _HOOK_MANAGER_REGISTRY.get(hook_manager_id)
        if hook_manager is None:
            msg = f"Hook manager {hook_manager_id} not found in registry"
            raise RuntimeError(msg)

        # Retrieve stored input
        tensor_input = hook_manager.inputs[layer_idx]

        # Cast input to match grad_output dtype for mixed precision training
        # This ensures all operations in _compute_compressed_grad work correctly
        tensor_input = tensor_input.to(grad_output.dtype)

        # Compute grad_input (needed for backprop to previous layers)
        # Cast weight to match grad_output dtype
        # (important for mixed precision training)
        # This mimics PyTorch's internal behavior in linear backward
        grad_input = grad_output @ weight.to(grad_output.dtype)

        # Compute compressed gradient directly (without full gradient)
        with torch.no_grad():
            compressed_grad = _compute_compressed_grad(
                grad_output=grad_output,
                tensor_input=tensor_input,
                has_bias=(bias is not None),
                compressor=hook_manager.compressors[layer_idx],
            )

            # Store compressed gradient in hook manager
            hook_manager.compressed_grads[layer_idx] = compressed_grad

            # Free input immediately
            hook_manager.inputs[layer_idx] = None

        # Return gradients for tensors:
        # - grad_input: needed for backprop
        # - None for weight: tells PyTorch NOT to compute weight.grad
        # - None for bias: tells PyTorch NOT to compute bias.grad
        # - None for hook_manager_id, layer_idx: not tensors
        return grad_input, None, None, None, None


def _compute_compressed_grad(
    grad_output: Tensor,
    tensor_input: Tensor,
    has_bias: bool,
    compressor: Compressor,
) -> Tensor:
    """Compute compressed gradient without materializing full gradient.

    Process:
    1. Add bias term to input if needed (gradient-specific preprocessing)
    2. Pass components to compressor which handles:
       - Sparsification of each component
       - Kronecker product computation
       - Final projection

    Args:
        grad_output: Gradient of the loss with respect to the output
        tensor_input: Input tensor to the layer
        has_bias: Whether the layer has a bias parameter
        compressor: Compressor object for gradient compression

    Returns:
        Compressed gradient tensor
    """
    # Gradient-specific preprocessing: Add bias column if needed
    if has_bias:
        # Determine if input is 3D (batch, seq, features) or 2D (batch, features)
        if tensor_input.dim() == 3:  # noqa: PLR2004 - Checking for 3D tensors
            batch_size, seq_length, _in_features = tensor_input.shape
            ones = torch.ones(
                batch_size,
                seq_length,
                1,
                device=tensor_input.device,
                dtype=tensor_input.dtype,
            )
        else:
            ones = torch.ones(
                tensor_input.size(0),
                1,
                device=tensor_input.device,
                dtype=tensor_input.dtype,
            )

        tensor_input = torch.cat([tensor_input, ones], dim=-1)

    # Delegate to compressor: it handles component-based compression
    return compressor.forward(grad_output, tensor_input)


class HookManager:
    """Hook manager that prevents full gradient materialization through monkey-patching.

    How it works:
    1. Replaces module.forward with a custom function that uses CompressedLinearBackward
    2. CompressedLinearBackward.backward() computes ONLY compressed gradients
    3. Returns None for weight/bias gradients, telling PyTorch to skip full computation
    4. Stores everything centrally in hook manager (memory efficient)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = "cpu",
        register_hooks: bool = True,
    ) -> None:
        """Initialize the hook manager.

        Args:
            model: The model to hook
            layer_names: Names of layers to hook (only Linear layers supported)
            device: Device for synchronization
            register_hooks: Whether to register hooks immediately
                (monkey-patch forward methods)
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device

        # Create mapping from layer name to index
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(layer_names)}

        # Create mapping from layer name to module
        self.layer_name_to_module = {}

        # Centralized storage arrays
        self.forward_hooks = [None] * len(layer_names)
        self.compressed_grads = [None] * len(layer_names)
        self.inputs = [None] * len(layer_names)

        # Unified compressors (handles all compression/decompression operations)
        self.compressors = [None] * len(layer_names)

        # Track hook registration status
        self.hooks_registered = False

        # Track whether hooks are enabled (can disable without unregistering)
        self.hooks_enabled = True

        # Register in global registry: Store ID, not self,
        # to avoid memory leaks in autograd graph
        self._hook_manager_id = id(self)
        _HOOK_MANAGER_REGISTRY[self._hook_manager_id] = self

        # Register hooks if requested
        if register_hooks:
            self._register_hooks()

        logger.info("Initialized HookManager with %d layers", len(layer_names))

    def _register_hooks(self) -> None:
        """Monkey-patch Linear layers to use our custom Function.

        This replaces module.forward with our custom function that prevents
        full gradient materialization.
        """
        if self.hooks_registered:
            logger.warning("Hooks already registered, skipping")
            return

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]

                # Cache the module
                self.layer_name_to_module[name] = module

                # Only support Linear layers
                if not isinstance(module, nn.Linear):
                    logger.warning("Layer %s is not nn.Linear, skipping", name)
                    continue

                # Save original forward method (so we can restore it later)
                module._original_forward = module.forward

                # Create wrapped forward that uses our custom Function
                wrapped_forward = functools.partial(
                    self._custom_linear_forward,
                    module,
                    idx,
                )

                # Replace the forward method (monkey-patching)
                module.forward = wrapped_forward

        self.hooks_registered = True
        logger.info("Successfully wrapped %d layers", len(self.layer_names))

    def _custom_linear_forward(
        self,
        module: nn.Linear,
        idx: int,
        module_input: Tensor,
    ) -> Tensor:
        """Replacement forward method that uses our custom Function.

        When requires_grad: Uses CompressedLinearBackward to prevent full
        gradient.
        Otherwise: Uses standard F.linear (no overhead)

        Args:
            module: The Linear module being hooked
            idx: Index of the layer in the layer_names list
            module_input: Input tensor to the layer

        Returns:
            Output tensor from the linear layer
        """
        if self.hooks_enabled:
            # Use our custom backward that computes only compressed gradients
            # Pass hook_manager_id (not self) to avoid keeping hook manager
            # in autograd graph
            return CompressedLinearBackward.apply(
                module_input,
                module.weight,
                module.bias,
                self._hook_manager_id,
                idx,
            )
        # Use original forward method when hooks are disabled
        # This ensures proper gradient computation for LoRA and other special layers
        if hasattr(module, "_original_forward"):
            return module._original_forward(module_input)
        # Fallback to F.linear if original forward not available
        return F.linear(module_input, module.weight, module.bias)

    def set_compressors(self, compressors: List[Compressor]) -> None:
        """Set unified compressor objects for each layer.

        Args:
            compressors: List of compressor objects, one for each hooked layer
        """
        self.compressors = compressors

    def enable_hooks(self) -> None:
        """Enable hooks to compute compressed gradients (default)."""
        self.hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable hooks to allow standard gradient computation."""
        self.hooks_enabled = False

    def get_compressed_grads(self) -> List[Tensor]:
        """Get all captured compressed gradients.

        Returns:
            List of compressed gradients for each hooked layer
        """
        return self.compressed_grads

    def remove_hooks(self) -> None:
        """Restore original forward methods and cleanup registry."""
        for module in self.layer_name_to_module.values():
            if hasattr(module, "_original_forward"):
                module.forward = module._original_forward
                delattr(module, "_original_forward")

        self.forward_hooks = [None] * len(self.layer_names)
        self.hooks_registered = False

        # Remove from registry
        if self._hook_manager_id in _HOOK_MANAGER_REGISTRY:
            del _HOOK_MANAGER_REGISTRY[self._hook_manager_id]

        logger.info("Restored original forward methods for all layers")

    def __del__(self) -> None:
        """Cleanup when hook manager is deleted."""
        if (
            hasattr(self, "_hook_manager_id")
            and self._hook_manager_id in _HOOK_MANAGER_REGISTRY
        ):
            del _HOOK_MANAGER_REGISTRY[self._hook_manager_id]
