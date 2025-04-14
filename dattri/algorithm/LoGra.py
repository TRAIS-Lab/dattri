from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, List

if TYPE_CHECKING:
    from typing import List, Optional, Union, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import functools

from tqdm import tqdm

from dattri.func.projection import random_project


class HookManager:
    """
    Manages hooks for efficient gradient component capturing and projection
    without requiring custom layer implementations.
    """
    def __init__(
            self,
            model: nn.Module,
            layer_names: List[str],
        ) -> None:
        """
        Initialize the hook manager

        Args:
            model: The model to hook
            layer_names: Names of layers to hook
        """
        self.model = model
        self.layer_names = layer_names

        # Create mapping from layer name to index for O(1) lookups
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(layer_names)}

        self.forward_hooks = [None] * len(layer_names)
        self.backward_hooks = [None] * len(layer_names)
        self.projected_grads = [None] * len(layer_names)
        self.inputs = [None] * len(layer_names)
        self.pre_activations = [None] * len(layer_names)
        self.normalized = [None] * len(layer_names)  # For LayerNorm
        self.projectors = [None] * len(layer_names)

        # Register hooks
        self._register_hooks()

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

    def set_projectors(self, projectors: List[Any]) -> None:
        """
        Set projector objects for each layer

        Args:
            projectors: List of projector objects, ordered by layer_names
        """
        self.projectors = projectors

    def get_projected_grads(self) -> List[Tensor]:
        """
        Get all captured projected gradients

        Returns:
            List of projected gradient tensors, ordered by layer_names
        """
        return self.projected_grads

    def get_projected_grad_by_name(self, name: str) -> Optional[Tensor]:
        """
        Get projected gradient for a specific layer by name

        Args:
            name: Layer name

        Returns:
            Projected gradient tensor for the specified layer
        """
        if name in self.layer_name_to_idx:
            idx = self.layer_name_to_idx[name]
            return self.projected_grads[idx]
        return None

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
                grad = self._linear_grad_from_grad_comp(
                    mod, idx, grad_pre_activation, per_sample=True
                )
            elif isinstance(mod, nn.LayerNorm):
                grad = self._layernorm_grad_from_grad_comp(
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
                self.projected_grads[idx] = grad.detach()

    def _linear_grad_from_grad_comp(
        self,
        layer: nn.Linear,
        idx: int,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute the gradient for Linear layers

        Args:
            layer: Linear layer
            idx: Layer index
            name: Layer name (kept for debugging)
            grad_pre_activation: Gradient of the pre-activation
            per_sample: Whether to compute per-sample gradients

        Returns:
            Projected gradient tensor
        """
        input_features = self.inputs[idx]
        is_3d = input_features.dim() == 3

        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            # Reshape 3D tensors to 2D for consistent processing
            input_features = input_features.reshape(-1, hidden_size)
            grad_pre_activation = grad_pre_activation.reshape(-1, layer.out_features)
        else:
            batch_size = input_features.shape[0]

        # Scale the gradient if we're computing per-sample gradients
        if per_sample:
            grad_pre_activation = grad_pre_activation * batch_size

        # Handle bias term by augmenting input with ones
        if layer.bias is not None:
            ones = torch.ones(
                input_features.size(0), 1,
                device=input_features.device,
                dtype=input_features.dtype
            )
            input_features = torch.cat([input_features, ones], dim=1)

        if is_3d:
            # Reshape back to 3D
            input_features = input_features.reshape(batch_size, seq_length, -1)
            grad_pre_activation = grad_pre_activation.reshape(batch_size, seq_length, -1)

        # Apply projectors if they exist
        projector = self.projectors[idx]
        if projector and hasattr(projector, 'projector_grad_comp') and projector.projector_grad_comp != (None, None):
            projector_grad_comp_1, projector_grad_comp_2 = projector.projector_grad_comp

            # Apply projection to gradient components
            grad_pre_activation_flatten = grad_pre_activation.view(-1, grad_pre_activation.shape[-1])
            input_features_flatten = input_features.view(-1, input_features.shape[-1])

            if is_3d:
                grad_pre_activation = projector_grad_comp_1(grad_pre_activation_flatten).view(
                    grad_pre_activation.shape[0], grad_pre_activation.shape[1], -1
                )
                input_features = projector_grad_comp_2(input_features_flatten).view(
                    input_features.shape[0], input_features.shape[1], -1
                )
            else:
                grad_pre_activation = projector_grad_comp_1(grad_pre_activation_flatten)
                input_features = projector_grad_comp_2(input_features_flatten)

        # Compute the outer product to get the gradient
        if is_3d:
            grad = torch.einsum('ijk,ijl->ikl', grad_pre_activation, input_features).reshape(batch_size, -1)
        else:
            grad = torch.einsum('bi,bj->bij', grad_pre_activation, input_features).reshape(batch_size, -1)

        # Apply final projector if available
        if projector and hasattr(projector, 'projector_grad') and projector.projector_grad is not None:
            grad = projector.projector_grad(grad)

        return grad

    def _layernorm_grad_from_grad_comp(
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
            name: Layer name (kept for debugging)
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

        # Apply projectors if they exist
        projector = self.projectors[idx]
        if projector and hasattr(projector, 'projector_grad_comp') and projector.projector_grad_comp != (None, None):
            projector_grad_comp_1, projector_grad_comp_2 = projector.projector_grad_comp
            grad_weight = projector_grad_comp_1(grad_weight)
            grad_bias = projector_grad_comp_2(grad_bias)

        # Concatenate weight and bias gradients
        grad = torch.cat((grad_weight, grad_bias), dim=1)

        # Apply final projector if available
        if projector and hasattr(projector, 'projector_grad') and projector.projector_grad is not None:
            grad = projector.projector_grad(grad)

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

def stable_inverse(matrix: torch.Tensor, damping: float = None) -> torch.Tensor:
    """
    Compute a numerically stable inverse of a matrix using eigendecomposition.

    Args:
        matrix: Input matrix to invert
        damping: Damping factor for numerical stability

    Returns:
        Stable inverse of the input matrix
    """
    # sometimes the matrix is a single number, so we need to check if it's a scalar
    if len(matrix.shape) == 0:
        if matrix == 0:
            # return a 2d 0 tensor
            return torch.tensor([[0.0]], device=matrix.device)
        else:
            if damping is None:
                return torch.tensor([[1.0 / (matrix * 1.1)]], device=matrix.device)
            else:
                return torch.tensor([[1.0 / (matrix * (1 + damping))]], device=matrix.device)

    # Add damping to the diagonal
    if damping is None:
        damping = 0.1 * torch.trace(matrix) / matrix.size(0)

    damped_matrix = matrix + damping * torch.eye(matrix.size(0), device=matrix.device)

    try:
        # Try Cholesky decomposition first (more stable)
        L = torch.linalg.cholesky(damped_matrix)
        inverse = torch.cholesky_inverse(L)
    except RuntimeError:
        print(f"Falling back to direct inverse due to Cholesky failure")
        # Fall back to direct inverse
        inverse = torch.inverse(damped_matrix)

    return inverse

class ProjectorContainer:
    """
    Container for projector functions associated with a layer.
    Used to store projectors without modifying the original layer.
    """
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index  # Add index field to identify position in array
        self.projector_grad = None
        self.projector_grad_comp = (None, None)


def setup_model_projectors(
    model: nn.Module,
    layer_names: List[str],
    projector_kwargs: Dict[str, Any],
    train_dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> List[ProjectorContainer]:
    """
    Sets up projectors for each layer in the model.

    Args:
        model: The PyTorch model
        layer_names: Names of layers to set projectors for
        projector_kwargs: Keyword arguments for projector configuration
        train_dataloader: DataLoader for training data (used to get input shapes)
        device: Device to run the model on

    Returns:
        List of projector containers, ordered by layer_names
    """
    if not projector_kwargs:
        return []

    # Extract configuration parameters
    proj_seed = projector_kwargs.get('proj_seed', 0)

    # Remove parameters that are handled separately
    kwargs_copy = projector_kwargs.copy()
    if 'proj_seed' in kwargs_copy:
        kwargs_copy.pop("proj_seed")

    # Initialize projector containers list
    projectors = [None] * len(layer_names)

    # Create name to index mapping for faster lookup
    name_to_index = {name: idx for idx, name in enumerate(layer_names)}

    # Run a forward pass to initialize model
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

    def capture_hook(name, mod, inp, out):
        layer_inputs[name] = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
        layer_outputs[name] = out

    # Register temporary hooks to capture layer I/O
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(lambda mod, inp, out, n=name: capture_hook(n, mod, inp, out))
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

            # Create appropriate projectors based on layer type
            if isinstance(module, nn.Linear):
                _setup_linear_projector(
                    projector,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs,
                )
            elif isinstance(module, nn.LayerNorm):
                _setup_layernorm_projector(
                    projector,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs,
                )
            else:
                raise ValueError(f"Unsupported layer type: {type(module)}")

            # Store the projector in the list at the correct index
            projectors[idx] = projector

    return projectors


def _setup_linear_projector(
    projector: ProjectorContainer,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    projector_kwargs: Dict[str, Any],
) -> None:
    """
    Set up projectors for a Linear layer

    Args:
        projector: ProjectorContainer to store the projectors
        layer: Linear layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        projector_kwargs: Keyword arguments for the projection
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

        ones = torch.ones(input_features.size(0), 1,
                         device=input_features.device,
                         dtype=input_features.dtype)
        input_features = torch.cat([input_features, ones], dim=1)

        if is_3d:
            input_features = input_features.reshape(batch_size, seq_length, -1)

    dumb_grad_comp_1 = torch.zeros_like(pre_activation.view(-1, pre_activation.shape[-1]))

    projector_grad_comp_1 = random_project(
        dumb_grad_comp_1,
        dumb_grad_comp_1.shape[0],
        proj_seed=base_seed,
        **projector_kwargs
    )

    dumb_grad_comp_2 = torch.zeros_like(input_features.view(-1, input_features.shape[-1]))
    projector_grad_comp_2 = random_project(
        dumb_grad_comp_2,
        dumb_grad_comp_2.shape[0],
        proj_seed=base_seed + 1,
        **projector_kwargs
    )

    projector.projector_grad_comp = (
        torch.compile(projector_grad_comp_1),
        torch.compile(projector_grad_comp_2)
    )


def _setup_layernorm_projector(
    projector: ProjectorContainer,
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    projector_kwargs: Dict[str, Any],
) -> None:
    """
    Set up projectors for a LayerNorm layer

    Args:
        projector: ProjectorContainer to store the projectors
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        projector_kwargs: Keyword arguments for the projection
    """
    if not layer.elementwise_affine:
        return

    if pre_activation is None:
        return

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
        torch.compile(projector_grad_comp_1),
        torch.compile(projector_grad_comp_2)
    )

class LoGraAttributor:
    """
    Optimized influence function calculator using hooks for efficient gradient projection.
    Works with standard PyTorch layers.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Union[str, List[str]],
        hessian: str = "raw",
        damping: float = None,
        device: str = 'cpu',
        cpu_offload: bool = False,
        projector_kwargs: Dict = None,
    ) -> None:
        """
        Optimized Influence Function Attributor.

        Args:
            model (nn.Module): PyTorch model.
            layer_names (List[str]): Names of layers to attribute.
            hessian (str): Type of Hessian approximation ("none", "raw", "kfac", "ekfac"). Defaults to "raw".
            damping (float): Damping used when calculating the Hessian inverse. Defaults to None.
            device (str): Device to run the model on. Defaults to 'cpu'.
            cpu_offload (bool): Whether to offload the model to CPU. Defaults to False.
            projector_kwargs (Dict): Keyword arguments for projector. Defaults to None.
        """
        self.model = model
        self.model.to(device)
        self.model.eval()

        # Ensure layer_names is a list
        if isinstance(layer_names, str):
            self.layer_names = [layer_names]
        else:
            self.layer_names = layer_names

        self.hessian = hessian
        self.damping = damping
        self.device = device
        self.cpu_offload = cpu_offload
        self.projector_kwargs = projector_kwargs or {}

        self.full_train_dataloader = None
        self.hook_manager = None
        self.cached_ifvp_train = None
        self.projectors = None

    def _setup_projectors(self, train_dataloader: torch.utils.data.DataLoader) -> None:
        """
        Set up projectors for the model layers

        Args:
            train_dataloader: DataLoader for training data
        """
        if not self.projector_kwargs:
            self.projectors = []
            return

        self.projectors = setup_model_projectors(
            self.model,
            self.layer_names,
            self.projector_kwargs,
            train_dataloader,
            self.device
        )

    def _calculate_ifvp(
        self,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> List[torch.Tensor]:
        """
        Compute FIM inverse Hessian vector product for each layer using hooks.

        Args:
            train_dataloader: DataLoader for training data

        Returns:
            List of tensors containing the FIM factors for each layer
        """
        # Set up the projectors
        if self.projectors is None:
            self._setup_projectors(train_dataloader)

        # Create name-to-index mapping for layer access
        layer_name_to_idx = {name: idx for idx, name in enumerate(self.layer_names)}

        # Initialize dynamic lists to store gradients for each layer
        per_layer_gradients = [[] for _ in self.layer_names]

        # Create hook manager
        self.hook_manager = HookManager(
            self.model,
            self.layer_names,
        )

        # Set projectors in the hook manager
        if self.projectors:
            self.hook_manager.set_projectors(self.projectors)

        # Iterate through the training data to compute gradients
        for train_batch_idx, train_batch in enumerate(tqdm(train_dataloader, desc="Processing training data")):
            # Zero gradients
            self.model.zero_grad()

            # Prepare inputs
            if isinstance(train_batch, dict):
                inputs = {k: v.to(self.device) for k, v in train_batch.items()}
            else:
                inputs = train_batch[0].to(self.device)

            # Forward pass
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            # Compute custom loss
            logp = -outputs.loss
            train_loss = logp - torch.log(1 - torch.exp(logp))

            # Backward pass
            train_loss.backward()

            # Get projected gradients from hook manager
            with torch.no_grad():
                projected_grads = self.hook_manager.get_projected_grads()

                # Collect gradients on device first (no immediate CPU transfer)
                for idx, grad in enumerate(projected_grads):
                    if grad is not None:
                        # Keep gradients on GPU and append to the list
                        per_layer_gradients[idx].append(grad.detach())

        # Remove hooks after collecting all gradients
        self.hook_manager.remove_hooks()

        # Process all collected gradients after the loop
        hessians = []
        train_grads = []

        # Calculate Hessian and prepare gradients for each layer
        for layer_idx in range(len(self.layer_names)):
            if per_layer_gradients[layer_idx]:
                # Concatenate all batches for this layer on GPU
                grads = torch.cat(per_layer_gradients[layer_idx], dim=0)

                # Compute Hessian on GPU (more efficient)
                hessian = torch.matmul(grads.t(), grads) / len(train_dataloader.sampler)

                # Store results based on offload preference
                if self.cpu_offload:
                    # Move data to CPU only once (not in the batch loop)
                    train_grads.append(grads.cpu())
                    hessians.append(hessian.cpu())
                else:
                    train_grads.append(grads)
                    hessians.append(hessian)
            else:
                train_grads.append(None)
                hessians.append(None)

        print(f"Computed gradient covariance for {len(self.layer_names)} modules")

        # Check if we have any valid gradients
        valid_grads = [grad is not None for grad in train_grads]
        if not any(valid_grads):
            print("Warning: No valid gradients were captured during the calculation.")
            print(f"Layer names: {self.layer_names}")
            return [torch.zeros(1) for _ in range(len(self.layer_names))]

        # Calculate inverse Hessian vector products
        if self.hessian == "none":
            return train_grads
        elif self.hessian == "raw":
            print("Computing gradient covariance inverse...")

            ifvp_train = []

            # Process each layer
            for layer_id, (grads, hessian) in enumerate(zip(train_grads, hessians)):
                if grads is None or hessian is None:
                    ifvp_train.append(None)
                    continue

                # Process Hessian inverse and IFVP calculation
                if self.cpu_offload:
                    # Move Hessian to GPU for inverse calculation
                    hessian_gpu = hessian.to(device=self.device)

                    # Calculate inverse on GPU (more efficient)
                    hessian_inv = stable_inverse(hessian_gpu, damping=self.damping)

                    # Process gradients in batches to avoid memory issues
                    batch_size = min(1024, grads.shape[0])  # Adjust based on available memory
                    results = []

                    for i in range(0, grads.shape[0], batch_size):
                        end_idx = min(i + batch_size, grads.shape[0])
                        grads_batch = grads[i:end_idx].to(device=self.device)

                        # Calculate IFVP for this batch
                        result_batch = torch.matmul(hessian_inv, grads_batch.t()).t()

                        # Move result back to CPU
                        results.append(result_batch.cpu())

                    # Combine results from all batches
                    ifvp_train.append(torch.cat(results, dim=0) if results else None)
                else:
                    # Calculate IFVP directly on GPU
                    hessian_inv = stable_inverse(hessian, damping=self.damping)
                    ifvp_train.append(torch.matmul(hessian_inv, grads.t()).t())

            print(f"Computed gradient covariance inverse for {len(self.layer_names)} modules")

            return ifvp_train
        else:
            raise ValueError(f"Unsupported Hessian approximation: {self.hessian}")

    def cache(
        self,
        full_train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        """
        Cache IFVP for the full training data.

        Args:
            full_train_dataloader: DataLoader for the full training data
        """
        print("Extracting information from training data...")
        self.full_train_dataloader = full_train_dataloader
        self.cached_ifvp_train = self._calculate_ifvp(full_train_dataloader)

    def attribute(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> torch.Tensor:
        """
        Attribute influence of training examples on test examples.

        Args:
            test_dataloader: DataLoader for test data
            train_dataloader: Optional DataLoader for training data if not cached

        Returns:
            Tensor of influence scores
        """
        if self.full_train_dataloader is not None and train_dataloader is not None:
            raise ValueError(
                "You have cached a training loader by .cache() and you are trying to attribute "
                "a different training loader. If this new training loader is a subset of the cached "
                "training loader, please don't input the training dataloader in .attribute() and "
                "directly use index to select the corresponding scores."
            )

        if train_dataloader is None and self.full_train_dataloader is None:
            raise ValueError(
                "You did not state a training loader in .attribute() and you did not cache a "
                "training loader by .cache(). Please provide a training loader or cache a "
                "training loader."
            )

        # Use cached IFVP or calculate new ones
        if train_dataloader is not None and self.full_train_dataloader is None:
            num_train = len(train_dataloader.sampler)
            ifvp_train = self._calculate_ifvp(train_dataloader)
        else:
            num_train = len(self.full_train_dataloader.sampler)
            ifvp_train = self.cached_ifvp_train

        # Storage device
        storage_device = "cpu" if self.cpu_offload else self.device

        # Initialize influence scores
        IF_score = torch.zeros(num_train, len(test_dataloader.sampler), device=storage_device)

        # Create hook manager for test examples
        self.hook_manager = HookManager(
            self.model,
            self.layer_names,
        )

        # Set projectors in the hook manager if available
        if self.projectors:
            self.hook_manager.set_projectors(self.projectors)

        # Collect test gradients first (similar to training data approach)
        per_layer_test_gradients = [[] for _ in self.layer_names]
        test_batch_indices = []

        # Process each test batch
        print("Collecting projected gradients from test data...")
        for test_batch_idx, test_batch in enumerate(tqdm(test_dataloader, desc="Collecting test gradients")):
            # Zero gradients
            self.model.zero_grad()

            # Prepare inputs
            if isinstance(test_batch, dict):
                inputs = {k: v.to(self.device) for k, v in test_batch.items()}
            else:
                inputs = test_batch[0].to(self.device)

            # Forward pass
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

            # Compute loss
            logp = -outputs.loss
            test_loss = logp - torch.log(1 - torch.exp(logp))

            # Backward pass
            test_loss.backward()

            # Get projected gradients from hook manager
            with torch.no_grad():
                projected_grads = self.hook_manager.get_projected_grads()

                # Store test batch indices for later mapping
                batch_size = test_batch[0].shape[0] if not isinstance(test_batch, dict) else next(iter(test_batch.values())).shape[0]
                col_st = test_batch_idx * batch_size
                col_ed = min(col_st + batch_size, len(test_dataloader.sampler))
                test_batch_indices.append((col_st, col_ed))

                # Collect test gradients
                for idx, grad in enumerate(projected_grads):
                    if grad is not None:
                        per_layer_test_gradients[idx].append(grad.detach())

            # Zero gradients for next iteration
            self.model.zero_grad()

        # Remove hooks
        self.hook_manager.remove_hooks()

        # Process influence scores in batches based on collected test gradients
        batch_size = min(64, len(test_batch_indices))  # Process multiple test batches at once

        for batch_start in range(0, len(test_batch_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(test_batch_indices))

            # Calculate influence for each layer
            for layer_id in range(len(self.layer_names)):
                if not per_layer_test_gradients[layer_id]:
                    continue

                if ifvp_train[layer_id] is None:
                    continue

                # Process test gradients for this batch
                for batch_idx in range(batch_start, batch_end):
                    test_grad = per_layer_test_gradients[layer_id][batch_idx]
                    col_st, col_ed = test_batch_indices[batch_idx]

                    # Compute influence scores
                    if self.cpu_offload:
                        # Move data to GPU in batches
                        ifvp_batch_size = min(1024, ifvp_train[layer_id].shape[0])

                        for i in range(0, ifvp_train[layer_id].shape[0], ifvp_batch_size):
                            end_idx = min(i + ifvp_batch_size, ifvp_train[layer_id].shape[0])

                            ifvp_batch = ifvp_train[layer_id][i:end_idx].to(device=self.device)
                            test_grad_gpu = test_grad.to(device=self.device)

                            # Compute partial influence
                            result = torch.matmul(ifvp_batch, test_grad_gpu.t())

                            # Update influence scores
                            IF_score_segment = IF_score[i:end_idx, col_st:col_ed].to(device=self.device)
                            IF_score_segment += result
                            IF_score[i:end_idx, col_st:col_ed] = IF_score_segment.cpu()
                    else:
                        # Compute on GPU directly
                        result = torch.matmul(ifvp_train[layer_id], test_grad.t())
                        IF_score[:, col_st:col_ed] += result

        # Return result
        return IF_score