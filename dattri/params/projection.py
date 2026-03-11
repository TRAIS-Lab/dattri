"""Pydantic models for projection parameters."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import BaseModel, ConfigDict, field_validator


class BaseProjectionParams(BaseModel):
    """Base projection params (no proj_dim).

    Args:
        proj_max_batch_size (int): The maximum batch size if the CudaProjector is
            used. Must be a multiple of 8. The maximum batch size is 32 for A100
            GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        proj_type (Literal["identity", "normal", "rademacher", "sjlt",
        "random_mask", "grass"]): The random projection type used for the projection.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    proj_max_batch_size: int = 64
    proj_seed: int = 0
    proj_type: Literal[
        "identity",
        "normal",
        "rademacher",
        "sjlt",
        "random_mask",
        "grass",
    ] = "normal"


class GeneralProjectionParams(BaseProjectionParams):
    """General projection params used by TracIn, TRAK, RandomProjectionParams.

    Args:
        proj_dim (int): Dimension of the projected feature.
        proj_max_batch_size (int): The maximum batch size if the CudaProjector is
            used. Must be a multiple of 8. The maximum batch size is 32 for A100
            GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        proj_type (Literal["identity", "normal", "rademacher", "sjlt",
        "random_mask", "grass"]): The random projection type used for the projection.
    """

    proj_dim: int


class LoGraProjectionParams(BaseProjectionParams):
    """Projection params for LoGra attributor.

    Args:
        proj_dim_per_layer (int): Dimension of the projected feature per layer.
        proj_max_batch_size (int): The maximum batch size if the CudaProjector is
            used. Must be a multiple of 8. The maximum batch size is 32 for A100
            GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        proj_type (Literal["identity", "normal", "rademacher", "sjlt",
        "random_mask", "grass"]): The random projection type used for the projection.
    """

    proj_dim_per_layer: int = 4096


class FactGrassProjectionParams(BaseProjectionParams):
    """Projection params for FactGraSS attributor.

    Args:
        proj_dim_per_layer (int): Dimension of the projected feature per layer.
        proj_max_batch_size (int): The maximum batch size if the CudaProjector is
            used. Must be a multiple of 8. The maximum batch size is 32 for A100
            GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        proj_type (Literal["identity", "normal", "rademacher", "sjlt",
        "random_mask", "grass"]): The random projection type used for the projection.
    """

    proj_dim_per_layer: int = 4096


class TracInProjectionParams(BaseProjectionParams):
    """Projection params for TracIn attributor.

    Args:
        proj_dim (int): Dimension of the projected feature.
        proj_max_batch_size (int): The maximum batch size if the CudaProjector is
            used. Must be a multiple of 8. The maximum batch size is 32 for A100
            GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        proj_type (Literal["identity", "normal", "rademacher", "sjlt",
        "random_mask", "grass"]): The random projection type used for the projection.
    """

    proj_dim: int = 512
    proj_max_batch_size: int = 32


class TRAKProjectionParams(GeneralProjectionParams):
    """Projection params for TRAK attributor.

    Args:
        proj_dim (int): Dimension of the projected feature.
        proj_max_batch_size (int): The maximum batch size if the CudaProjector is
            used. Must be a multiple of 8. The maximum batch size is 32 for A100
            GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        proj_type (Literal["identity", "normal", "rademacher", "sjlt",
        "random_mask", "grass"]): The random projection type used for the projection.
    """

    proj_dim: int = 512
    proj_max_batch_size: int = 32


class DVEmbProjectionParams(BaseProjectionParams):
    """Projection params for DVEmb; proj_dim_per_layer can be None for no projection.

    Args:
        proj_dim_per_layer (int): Dimension of the projected feature per layer.
        proj_max_batch_size (int): The maximum batch size if the CudaProjector is
            used. Must be a multiple of 8. The maximum batch size is 32 for A100
            GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        proj_type (Literal["identity", "normal", "rademacher", "sjlt",
        "random_mask", "grass"]): The random projection type used for the projection.
    """

    proj_dim_per_layer: Optional[int] = None
    proj_type: Literal[
        "identity",
        "normal",
        "rademacher",
        "sjlt",
        "random_mask",
        "grass",
    ] = "identity"


class RandomProjectionParams(GeneralProjectionParams):
    """Params for random_project().

    Args:
        proj_dim (int): Dimension of the projected feature.
        proj_max_batch_size (int): The maximum batch size if the CudaProjector is
            used. Must be a multiple of 8. The maximum batch size is 32 for A100
            GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        feature_batch_size (int): The batch size of each tensor in the feature
            about to be projected. The typical type of feature are gradients of
            torch.nn.Module model but can be restricted to this.
        device (torch.device): Device to use. Defaults to cpu.
        proj_type (Literal["identity", "normal", "rademacher", "sjlt",
        "random_mask", "grass"]): The random projection type used for the projection.
        device (Union[str, torch.device]): "cuda" or "cpu". Defaults to "cpu".
    """

    proj_dim: int
    proj_max_batch_size: int
    proj_seed: int
    feature_batch_size: int
    device: torch.device
    proj_type: Literal[
        "identity",
        "normal",
        "rademacher",
        "sjlt",
        "random_mask",
        "grass",
    ] = "normal"

    @field_validator("device", mode="before")
    @classmethod
    def _device_to_torch(cls, v: object) -> torch.device:
        if isinstance(v, torch.device):
            return v
        return torch.device(str(v))
