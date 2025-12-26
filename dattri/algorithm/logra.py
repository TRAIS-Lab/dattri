"""LoGra Attributor - Influence Function with block-diagonal projections.

Kronecker-factored projections for efficient gradient compression.

This is a thin wrapper around BlockProjectedIFAttributor that configures
it for the LoGra method.

LoGra uses:
- Sparsifier: normal projection (factorized)
- Projector: identity (no second stage compression)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Literal, Optional, Union

if TYPE_CHECKING:
    from dattri.task import AttributionTask

from dattri.algorithm.block_projected_if.block_projected_if import (
    BlockProjectedIFAttributor,
)


class LoGraAttributor(BlockProjectedIFAttributor):
    """LoGra Attributor.

    Low-Rank Gradient Projection (LoGra) attributor that uses normal projection
    for the first stage and identity projection (no compression) for the second
    stage.

    This is equivalent to the original LoGra method from the paper.

    The projection is factorized: if you specify proj_dim=4096, each component
    will have dimension sqrt(4096)=64, and the Kronecker product will have
    dimension 4096.
    """

    def __init__(
        self,
        task: "AttributionTask",
        layer_names: Optional[Union[str, List[str]]] = None,
        hessian: Literal["Identity", "eFIM"] = "eFIM",
        damping: Optional[float] = None,
        device: str = "cpu",
        proj_dim: int = 4096,
        offload: Literal["none", "cpu", "disk"] = "cpu",
        cache_dir: Optional[str] = None,
        chunk_size: int = 16,
    ) -> None:
        """Initialize LoGra attributor.

        Args:
            task: Attribution task containing model, loss function, and checkpoints
            layer_names: Names of layers where gradients will be collected.
                If None, uses all Linear layers.
            hessian: Type of Hessian approximation ("Identity", "eFIM").
            damping: Damping factor for Hessian inverse (when hessian="eFIM")
            device: Device to run computations on
            proj_dim: Projection dimension after Kronecker product (default:
                4096). Must be a perfect square. The per-component dimension
                will be √proj_dim. For example, proj_dim=4096 gives
                per-component dim of 64.
            offload: Memory management strategy ("none", "cpu", "disk")
            cache_dir: Directory for caching (required when offload="disk")
            chunk_size: Chunk size for processing in disk offload

        Raises:
            ValueError: If proj_dim is not a perfect square.
        """
        # Validate that proj_dim is a perfect square
        sqrt_proj_dim = int(math.sqrt(proj_dim))
        if sqrt_proj_dim * sqrt_proj_dim != proj_dim:
            msg = (
                "proj_dim must be a perfect square for factorized projection. "
                f"Got {proj_dim}, but sqrt({proj_dim}) = "
                f"{math.sqrt(proj_dim)} is not an integer."
            )
            raise ValueError(msg)

        # Compute per-component dimension
        per_component_dim = sqrt_proj_dim

        # Set LoGra-specific configuration
        sparsifier_kwargs = {
            "device": device,
            "proj_dim": per_component_dim,
            "proj_max_batch_size": 64,
            "proj_type": "normal",
        }

        projector_kwargs = {
            "device": device,
            "proj_dim": -1,  # -1 means no compression (identity)
            "proj_max_batch_size": 64,
            "proj_type": "identity",
        }

        # Initialize the base class with LoGra-specific configuration
        super().__init__(
            task=task,
            layer_names=layer_names,
            hessian=hessian,
            damping=damping,
            device=device,
            sparsifier_kwargs=sparsifier_kwargs,
            projector_kwargs=projector_kwargs,
            offload=offload,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
        )
