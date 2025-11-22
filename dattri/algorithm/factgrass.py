"""FactGraSS Attributor - Factorized Gradient Sparsification and Sparse Projection.

This is a thin wrapper around BlockProjectedIFAttributor that configures
it for the FactGraSS method.

FactGraSS uses:
- Sparsifier: random_mask projection (first stage, factorized)
- Projector: sjlt projection (second stage)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Literal, Optional, Union

if TYPE_CHECKING:
    from dattri.task import AttributionTask

from dattri.algorithm.block_projected_if.block_projected_if import (
    BlockProjectedIFAttributor,
)


class FactGraSSAttributor(BlockProjectedIFAttributor):
    """FactGraSS Attributor.

    Factorized Gradient Sketching with Structured Sparsity (FactGraSS)
    attributor that uses random mask projection for the first stage and SJLT
    (Sparse Johnson-Lindenstrauss Transform) for the second stage.

    This is a follow-up work to LoGra that provides better compression by
    using a two-stage compression pipeline.

    The first stage is factorized with a blowup factor:
    - If proj_dim=4096 and blowup_factor=4:
      - Intermediate dimension = 4096 * 4 = 16384
      - Per-component dimension = sqrt(16384) = 128
      - After second stage: final dimension = 4096
    """

    def __init__(
        self,
        task: "AttributionTask",
        layer_names: Optional[Union[str, List[str]]] = None,
        hessian: Literal["Identity", "eFIM"] = "eFIM",
        damping: Optional[float] = None,
        device: str = "cpu",
        proj_dim: int = 4096,
        blowup_factor: int = 4,
        offload: Literal["none", "cpu", "disk"] = "cpu",
        cache_dir: Optional[str] = None,
        chunk_size: int = 16,
    ) -> None:
        """Initialize FactGraSS attributor.

        Args:
            task: Attribution task containing model, loss function, and checkpoints
            layer_names: Names of layers where gradients will be collected.
                If None, uses all Linear layers.
            hessian: Type of Hessian approximation ("Identity", "eFIM").
            damping: Damping factor for Hessian inverse (when hessian="eFIM")
            device: Device to run computations on
            proj_dim: Projection dimension after second stage (default: 4096).
            blowup_factor: Multiplier for intermediate dimension after
                sparsification (default: 4). The intermediate dimension will be
                proj_dim * blowup_factor, which must be a perfect square for
                factorized projection. For example, proj_dim=4096 and
                blowup_factor=4 gives intermediate_dim=16384, per-component
                dim=128.
            offload: Memory management strategy ("none", "cpu", "disk")
            cache_dir: Directory for caching (required when offload="disk")
            chunk_size: Chunk size for processing in disk offload

        Raises:
            ValueError: If intermediate_dim (proj_dim * blowup_factor) is not
                a perfect square.
        """
        # Compute intermediate dimension after first stage
        intermediate_dim = proj_dim * blowup_factor

        # Validate that intermediate_dim is a perfect square
        sqrt_intermediate_dim = int(math.sqrt(intermediate_dim))
        if sqrt_intermediate_dim * sqrt_intermediate_dim != intermediate_dim:
            msg = (
                "intermediate_dim (proj_dim * blowup_factor) must be a "
                "perfect square for factorized projection. Got "
                f"proj_dim={proj_dim}, blowup_factor={blowup_factor}, "
                f"intermediate_dim={intermediate_dim}, but "
                f"sqrt({intermediate_dim}) = {math.sqrt(intermediate_dim)} "
                "is not an integer."
            )
            raise ValueError(msg)

        # Compute per-component dimension for the sparsifier
        per_component_dim = sqrt_intermediate_dim

        # Set FactGraSS-specific configuration
        sparsifier_kwargs = {
            "device": device,
            "proj_dim": per_component_dim,
            "proj_max_batch_size": 64,
            "proj_type": "random_mask",
        }

        projector_kwargs = {
            "device": device,
            "proj_dim": proj_dim,
            "proj_max_batch_size": 64,
            "proj_type": "sjlt" if device == "cpu" else "normal",
        }

        # Initialize the base class with FactGraSS-specific configuration
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
