"""Pydantic models for projection parameters."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import BaseModel, ConfigDict, field_validator


class BaseProjectionParams(BaseModel):
    """Base projection params (no proj_dim)."""

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
    """General projection params used by TracIn, TRAK, RandomProjectionParams."""

    proj_dim: int


class LoGraProjectionParams(BaseProjectionParams):
    """Projection params for LoGra attributor."""

    proj_dim_per_layer: int = 4096


class FactGrassProjectionParams(BaseProjectionParams):
    """Projection params for FactGraSS attributor."""

    proj_dim_per_layer: int = 4096


class TracInProjectionParams(BaseProjectionParams):
    """Projection params for TracIn attributor."""

    proj_dim: int = 512
    proj_max_batch_size: int = 32


class TRAKProjectionParams(GeneralProjectionParams):
    """Projection params for TRAK attributor."""

    proj_dim: int = 512
    proj_max_batch_size: int = 32


class DVEmbProjectionParams(BaseProjectionParams):
    """Projection params for DVEmb; proj_dim_per_layer can be None for no projection."""

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
    """Params for random_project(); adds feature_batch_size and device."""

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
