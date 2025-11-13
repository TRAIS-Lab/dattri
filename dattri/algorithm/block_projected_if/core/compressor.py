"""Compression container classes for two-stage tensor compression.

This module implements a two-stage compression pipeline:
1. Sparsifier: Factorized component-wise compression (always factorized)
2. Projector: Final projection after sparsification (always non-factorized)
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple

import logging

import torch
from torch import Tensor, nn

from dattri.func.projection import AbstractProjector, random_project

# Configure logger
logger = logging.getLogger(__name__)


class ProjectionContainer(ABC):  # noqa: B024 - Abstract base class, forward method defined in subclasses
    """Abstract base class for compression containers."""

    def __init__(self, name: str, index: int) -> None:
        """Initialize the projection container.

        Args:
            name: Name of the layer.
            index: Index of the layer.
        """
        self.name = name
        self.index = index


class Sparsifier(ProjectionContainer):
    """Container for sparsification (first stage of two-step compression).

    Sparsification is always factorized, operating component-wise:
        Pv = (P_1 ⊗ P_2) (v1 ⊗ v2) = (P_1 v1) ⊗ (P_2 v2),
    with P_1 in R^{d_1 * k_1'}, P_2 in R^{d_2 * k_2'},
    and v in R^{d_1 * d_2}.
    """

    def __init__(self, name: str, index: int) -> None:
        """Initialize the sparsifier.

        Args:
            name: Name of the layer.
            index: Index of the layer.
        """
        super().__init__(name, index)
        # Sparsifier functions (always factorized)
        self.sparsifier_comp: Tuple[
            AbstractProjector,
            AbstractProjector,
        ] = (None, None)

        # Dimensions after sparsification
        self.intermediate_dims = None  # (k_1', k_2')

    def forward(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply sparsification to input components.

        Sparsification is always factorized, operating component-wise:
            Input: (v1, v2) - input components [batch, d_1] and
            [batch, d_2]
            Output: (v1_sparse, v2_sparse) - sparsified components
            [batch, k_1'] and [batch, k_2']

        Use case: During gradient computation, apply sparsifiers to each
        component (grad_output, input)

        Args:
            v1: First input component [batch, d_1]
            v2: Second input component [batch, d_2]

        Returns:
            (component_1_sparse, component_2_sparse) - sparsified components
        """
        sparsifier_1, sparsifier_2 = self.sparsifier_comp

        # Apply sparsifiers component-wise
        v1_sparse = sparsifier_1(v1, ensemble_id=0)
        v2_sparse = sparsifier_2(v2, ensemble_id=0)

        return v1_sparse, v2_sparse


class Projector(ProjectionContainer):
    """Container for projection (second stage of two-step compression).

    Projection is a simple, one-step projection:
        Pv = P v.
    with P in R^{k * k'} and v in R^{k'}.
    """

    def __init__(self, name: str, index: int) -> None:
        """Initialize the projector.

        Args:
            name: Name of the layer.
            index: Index of the layer.
        """
        super().__init__(name, index)
        self.projector: AbstractProjector = None

    def forward(self, v: Tensor) -> Tensor:
        """Apply final projection to intermediate tensor after sparsification.

        Args:
            v: batch vector [batch, k'] after sparsification

        Returns:
            Compressed batch vector [batch, k] after projection
        """
        return self.projector(v, ensemble_id=0)


class Compressor(ProjectionContainer):
    """Unified container for full two-stage compression pipeline.

    Forward:   input → Sparsifier → Projector → compressed
    """

    def __init__(self, name: str, index: int) -> None:
        """Initialize the compressor.

        Args:
            name: Name of the layer.
            index: Index of the layer.
        """
        super().__init__(name, index)
        self.sparsifier: Sparsifier = None
        self.projector: Projector = None

    @torch.compile
    def forward(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> torch.Tensor:
        """Apply full compression pipeline.

        Pipeline: input → sparsifier → Kronecker product → projector → output

        Component-based compression:
           - Components can be 2D [batch, features] or 3D
             [batch, seq, features]
           - Applies sparsification to each component (grad_output, input)
           - Computes Kronecker product
           - Applies projection

        Args:
            v1: First input component [batch, d_1] or [batch, seq, d_1]
            v2: Second input component [batch, d_2] or [batch, seq, d_2]

        Returns:
            Compressed output tensor
        """
        # Detect if components are 3D or 2D
        is_3d = v2.dim() == 3  # noqa: PLR2004 - Checking for 3D tensors

        if is_3d:
            # Extract dimensions for 3D case
            batch_size, seq_length, _ = v2.shape

            # Reshape to 2D for sparsification
            v1_2d = v1.reshape(-1, v1.shape[-1])
            v2_2d = v2.reshape(-1, v2.shape[-1])
        else:
            # Already 2D
            batch_size = v2.shape[0]
            v1_2d = v1
            v2_2d = v2

        # Stage 1: Apply sparsification to each component
        v1_sparse, v2_sparse = self.sparsifier.forward(
            v1_2d,
            v2_2d,
        )

        # Stage 1.5: Compute Kronecker product (outer product)
        if is_3d:
            # Reshape back to 3D for proper outer product computation
            v1_sparse_3d = v1_sparse.reshape(batch_size, seq_length, -1)
            v2_sparse_3d = v2_sparse.reshape(batch_size, seq_length, -1)

            # Compute outer product with batch scaling
            outer_product = torch.einsum(
                "bsi,bsj->bij",
                v1_sparse_3d * batch_size,
                v2_sparse_3d,
            )
        else:
            # Compute outer product for 2D case with batch scaling
            outer_product = torch.einsum(
                "bi,bj->bij",
                v1_sparse * batch_size,
                v2_sparse,
            )

        # Flatten the outer product result
        intermediate = outer_product.reshape(batch_size, -1)

        # Stage 2: Apply projection
        return self.projector.forward(intermediate)


def setup_model_compressors(  # noqa: PLR0912, PLR0914, PLR0915 - Complex setup with many branches, variables, and statements for handling different layer types
    model: nn.Module,
    layer_names: List[str],
    sparsifier_kwargs: Optional[Dict[str, Any]] = None,
    projector_kwargs: Optional[Dict[str, Any]] = None,
    sample_inputs: Optional[Dict[str, Tensor]] = None,
    device: str = "cpu",
) -> List[Compressor]:
    """Set up unified Compressors for each layer in the model.

    Each Compressor encapsulates both sparsification and projection stages.

    Args:
        model: The PyTorch model
        layer_names: Names of layers to set compressors for
        sparsifier_kwargs: Keyword arguments for sparsifier config (optional)
        projector_kwargs: Keyword arguments for projector config (optional)
        sample_inputs: Input batch to run a forward pass (optional)
        device: Device to run the model on

    Raises:
        ValueError: If unsupported layer types are encountered.

    Returns:
        List of Compressors, ordered by layer_names
    """
    if sample_inputs is None:
        return []

    # Initialize compressors list
    compressors = [None] * len(layer_names)
    if not (sparsifier_kwargs or projector_kwargs):
        return compressors

    # Create name to index mapping for faster lookup
    name_to_index = {name: idx for idx, name in enumerate(layer_names)}

    # Extract configuration parameters
    sparsifier_seed = sparsifier_kwargs.get("proj_seed", 0) if sparsifier_kwargs else 0
    projector_seed = projector_kwargs.get("proj_seed", 0) if projector_kwargs else 0

    # Remove parameters that are handled separately
    if sparsifier_kwargs:
        sparsifier_kwargs_copy = sparsifier_kwargs.copy()
        if "proj_seed" in sparsifier_kwargs_copy:
            sparsifier_kwargs_copy.pop("proj_seed")
    else:
        sparsifier_kwargs_copy = {}

    if projector_kwargs:
        projector_kwargs_copy = projector_kwargs.copy()
        if "proj_seed" in projector_kwargs_copy:
            projector_kwargs_copy.pop("proj_seed")
    else:
        projector_kwargs_copy = {}

    # Ensure model is on the correct device before running forward pass
    original_device = next(model.parameters()).device
    if str(original_device) != device:
        model.to(device)

    # Use no_grad to avoid autograd issues during setup
    with torch.no_grad():
        if isinstance(sample_inputs, dict):
            inputs = {k: v.to(device) for k, v in sample_inputs.items()}
            model(**inputs)
        else:
            inputs = sample_inputs[0].to(device)
            model(inputs)

    # First, capture inputs and outputs for each layer
    layer_inputs = {}
    layer_outputs = {}
    hooks = []

    def capture_hook(  # noqa: ANN202 - Hook signature matches PyTorch API
        name: str,
        mod: nn.Module,  # noqa: ARG001 - Required by PyTorch hook API
        inp: tuple,
        out: Tensor,
    ):
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
    with torch.no_grad():
        if isinstance(sample_inputs, dict):
            model(**inputs)
        else:
            model(inputs)

    # Remove temporary hooks
    for hook in hooks:
        hook.remove()

    # Create sparsifiers and projectors for each layer
    module_list = list(model.named_modules())

    for module_id, (module_name, module) in enumerate(module_list):
        if module_name in layer_names:
            idx = name_to_index[module_name]

            # Create unified compressor container
            compressor = Compressor(module_name, idx)

            # Create sparsifier (ALWAYS factorized) - Stage 1
            sparsifier = Sparsifier(module_name, idx)
            base_seed = sparsifier_seed + int(1e4) * module_id
            sparse_kwargs = sparsifier_kwargs_copy.copy()

            # Create appropriate sparsifiers based on layer type
            # Sparsifiers are ALWAYS factorized
            if isinstance(module, nn.Linear):
                _setup_linear_sparsifier(
                    sparsifier,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    sparse_kwargs,
                )
            elif isinstance(module, nn.LayerNorm):
                _setup_layernorm_sparsifier(
                    sparsifier,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    sparse_kwargs,
                )
            else:
                msg = f"Unsupported layer type: {type(module)}"
                raise ValueError(msg)

            # Check if sparsifier was successfully initialized
            if sparsifier.sparsifier_comp == (None, None):
                logger.warning(
                    "Skipping layer %s: sparsifier setup failed "
                    "(likely missing inputs/outputs)",
                    module_name,
                )
                continue

            # Create projector
            # (ALWAYS non-factorized, operates after sparsification) - Stage 2
            projector = Projector(module_name, idx)
            base_seed = projector_seed + int(1e4) * module_id + 1
            proj_kwargs = projector_kwargs_copy.copy()

            # Set up projectors for sparsified dimensions
            # Projectors are ALWAYS non-factorized and operate AFTER
            # sparsification
            if isinstance(module, nn.Linear):
                _setup_linear_projector(
                    projector,
                    sparsifier,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs,
                )
            elif isinstance(module, nn.LayerNorm):
                _setup_layernorm_projector(
                    projector,
                    sparsifier,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs,
                )
            else:
                msg = f"Unsupported layer type: {type(module)}"
                raise ValueError(msg)

            # Check if projector was successfully initialized
            if projector.projector is None:
                logger.warning("Skipping layer %s: projector setup failed", module_name)
                continue

            # Assemble compressor container with both stages
            compressor.sparsifier = sparsifier
            compressor.projector = projector

            compressors[idx] = compressor

    return compressors


def _setup_linear_sparsifier(
    sparsifier: Sparsifier,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any],
) -> None:
    """Set up sparsifier for a Linear layer.

    STRICT CONVENTION:
    - Sparsifier MUST contain factorized sparsifiers
    - Sparsifiers operate BEFORE outer product on components

    Args:
        sparsifier: Sparsifier to store the sparsifier functions
        layer: Linear layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if pre_activation is None or layer_input is None:
        return

    batch_size = pre_activation.shape[0]
    is_3d = layer_input.dim() == 3  # noqa: PLR2004 - Checking for 3D tensors

    input_features = layer_input
    if layer.bias is not None:
        if is_3d:
            batch_size, seq_length, _hidden_size = input_features.shape
            input_features = input_features.reshape(-1, _hidden_size)
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

    # Sparsifiers are ALWAYS factorized
    sample_comp_1 = torch.zeros_like(pre_activation.view(-1, pre_activation.shape[-1]))

    # For identity projection, use feature dimension instead of -1
    proj_dim_comp_1 = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim_comp_1 == -1:
        proj_dim_comp_1 = sample_comp_1.shape[-1]

    sparsifier_comp_1 = random_project(
        sample_comp_1,
        sample_comp_1.shape[0],
        proj_dim=proj_dim_comp_1,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu"),
    )

    sample_comp_2 = torch.zeros_like(input_features.view(-1, input_features.shape[-1]))

    # For identity projection, use feature dimension instead of -1
    proj_dim_comp_2 = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim_comp_2 == -1:
        proj_dim_comp_2 = sample_comp_2.shape[-1]

    _sparsifier_comp_2 = random_project(
        sample_comp_2,
        sample_comp_2.shape[0],
        proj_dim=proj_dim_comp_2,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed + 1,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu"),
    )

    # Store dimensions needed for transpose operation
    # Test projection to get actual intermediate dimensions
    test_comp_1 = sparsifier_comp_1(sample_comp_1[:1], ensemble_id=0)
    test_comp_2 = _sparsifier_comp_2(sample_comp_2[:1], ensemble_id=0)
    sparsifier.intermediate_dims = (
        test_comp_1.shape[-1],
        test_comp_2.shape[-1],
    )

    # Store factorized sparsifiers in sparsifier_comp
    sparsifier.sparsifier_comp = (
        sparsifier_comp_1,
        _sparsifier_comp_2,
    )


def _setup_linear_projector(  # noqa: PLR0914 - Complex projector setup requires many local variables for dimension calculations
    projector: Projector,
    sparsifier: Sparsifier,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any],
) -> None:
    """Set up projector for a Linear layer after sparsification.

    STRICT CONVENTION:
    - Projector MUST contain non-factorized projectors
    - Projectors operate AFTER outer product on flattened gradient
    - Sparsifiers (in Sparsifier) are ALWAYS factorized

    Args:
        projector: Projector to store the projector
        sparsifier: Sparsifier with sparsification functions
        layer: Linear layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if pre_activation is None or layer_input is None:
        return

    batch_size = pre_activation.shape[0]
    is_3d = layer_input.dim() == 3  # noqa: PLR2004 - Checking for 3D tensors

    # Extract sparsifier components
    sparsifier_comp_1, _sparsifier_comp_2 = sparsifier.sparsifier_comp

    # Get sample tensors to determine output dimensions of sparsifiers
    if is_3d:
        sample_pre_activation = pre_activation[:1, :1].reshape(
            -1,
            pre_activation.shape[-1],
        )
        sparse_sample_pre_activation = sparsifier_comp_1(
            sample_pre_activation,
            ensemble_id=0,
        )
        sparsified_dim_1 = sparse_sample_pre_activation.shape[-1]

        input_features = layer_input
        if layer.bias is not None:
            batch_size, seq_length, _hidden_size = input_features.shape
            input_features_with_bias = torch.cat(
                [
                    input_features,
                    torch.ones(
                        batch_size,
                        seq_length,
                        1,
                        device=input_features.device,
                        dtype=input_features.dtype,
                    ),
                ],
                dim=2,
            )
        else:
            input_features_with_bias = input_features

        sample_input_features = input_features_with_bias[:1, :1].reshape(
            -1,
            input_features_with_bias.shape[-1],
        )
        sparse_sample_input_features = _sparsifier_comp_2(
            sample_input_features,
            ensemble_id=0,
        )
        sparsified_dim_2 = sparse_sample_input_features.shape[-1]
    else:
        sample_pre_activation = pre_activation[:1]
        sparse_sample_pre_activation = sparsifier_comp_1(
            sample_pre_activation,
            ensemble_id=0,
        )
        sparsified_dim_1 = sparse_sample_pre_activation.shape[-1]

        input_features = layer_input
        if layer.bias is not None:
            input_features_with_bias = torch.cat(
                [
                    input_features,
                    torch.ones(
                        input_features.size(0),
                        1,
                        device=input_features.device,
                        dtype=input_features.dtype,
                    ),
                ],
                dim=1,
            )
        else:
            input_features_with_bias = input_features

        sample_input_features = input_features_with_bias[:1]
        sparse_sample_input_features = _sparsifier_comp_2(
            sample_input_features,
            ensemble_id=0,
        )
        sparsified_dim_2 = sparse_sample_input_features.shape[-1]

    # Create non-factorized projector (operates on flattened intermediate
    # after sparsification). Calculate dimension after sparsification.
    intermediate_dim = sparsified_dim_1 * sparsified_dim_2

    sample_intermediate = torch.zeros(
        (batch_size, intermediate_dim),
        device=pre_activation.device,
        dtype=pre_activation.dtype,
    )

    # For identity projection, use feature dimension instead of -1
    proj_dim = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim == -1:
        proj_dim = intermediate_dim

    projector_func = random_project(
        sample_intermediate,
        sample_intermediate.shape[0],
        proj_dim=proj_dim,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu"),
    )

    # Store in projector attribute
    projector.projector = projector_func


def _setup_layernorm_sparsifier(
    sparsifier: Sparsifier,
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any],
) -> None:
    """Set up sparsifier for a LayerNorm layer.

    STRICT CONVENTION:
    - Sparsifier MUST contain factorized sparsifiers
    - Sparsifiers operate BEFORE outer product on components

    Args:
        sparsifier: Sparsifier to store the sparsifier functions
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if not layer.elementwise_affine or layer_input is None:
        return

    # Sparsifiers are ALWAYS factorized
    sample_comp_1 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))

    # For identity projection, use feature dimension instead of -1
    proj_dim_comp_1 = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim_comp_1 == -1:
        proj_dim_comp_1 = sample_comp_1.shape[-1]

    sparsifier_comp_1 = random_project(
        sample_comp_1,
        sample_comp_1.shape[0],
        proj_dim=proj_dim_comp_1,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu"),
    )

    sample_comp_2 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))

    # For identity projection, use feature dimension instead of -1
    proj_dim_comp_2 = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim_comp_2 == -1:
        proj_dim_comp_2 = sample_comp_2.shape[-1]

    _sparsifier_comp_2 = random_project(
        sample_comp_2,
        sample_comp_2.shape[0],
        proj_dim=proj_dim_comp_2,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed + 1,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu"),
    )

    # Store dimensions needed for transpose operation
    # Test projection to get actual intermediate dimensions
    test_comp_1 = sparsifier_comp_1(sample_comp_1[:1], ensemble_id=0)
    test_comp_2 = _sparsifier_comp_2(sample_comp_2[:1], ensemble_id=0)
    sparsifier.intermediate_dims = (
        test_comp_1.shape[-1],
        test_comp_2.shape[-1],
    )

    # Store factorized sparsifiers in sparsifier_comp
    sparsifier.sparsifier_comp = (
        sparsifier_comp_1,
        _sparsifier_comp_2,
    )


def _setup_layernorm_projector(
    projector: Projector,
    sparsifier: Sparsifier,
    layer: nn.LayerNorm,
    layer_input: Tensor,  # noqa: ARG001 - Required for API consistency with _setup_linear_projector
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any],
) -> None:
    """Set up projector for a LayerNorm layer after sparsification.

    STRICT CONVENTION:
    - Projector MUST contain non-factorized projectors
    - Projectors operate AFTER outer product on flattened gradient
    - Sparsifiers (in Sparsifier) are ALWAYS factorized

    Args:
        projector: Projector to store the projector
        sparsifier: Sparsifier with sparsification functions
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if not layer.elementwise_affine or pre_activation is None:
        return

    # Apply sparsification to get correct dimensions for projector setup
    sparsifier_comp_1, _sparsifier_comp_2 = sparsifier.sparsifier_comp

    # Get sample tensors to determine output dimensions of sparsifiers
    sample_pre_activation = pre_activation[:1]
    sparse_sample_pre_activation = sparsifier_comp_1(
        sample_pre_activation,
        ensemble_id=0,
    )

    # Create non-factorized projector (operates on concatenated sparsified components)
    # For LayerNorm: intermediate is concatenation of weight and bias components
    intermediate_dim = sparse_sample_pre_activation.shape[-1] * 2  # weight + bias

    sample_intermediate = torch.zeros(
        (pre_activation.shape[0], intermediate_dim),
        device=pre_activation.device,
        dtype=pre_activation.dtype,
    )

    # For identity projection, use feature dimension instead of -1
    proj_dim = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim == -1:
        proj_dim = intermediate_dim

    projector_func = random_project(
        sample_intermediate,
        sample_intermediate.shape[0],
        proj_dim=proj_dim,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu"),
    )

    # Store in projector attribute
    projector.projector = projector_func
