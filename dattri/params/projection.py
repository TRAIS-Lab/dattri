"""Pydantic models for projection parameters."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import BaseModel, ConfigDict, field_validator


class ProjectionParams(BaseModel):
    """Base projection parameters."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    proj_dim: int
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


class GeneralProjectionParams(BaseModel):
    """General projection params used by LoGra, FactGraSS, TracIn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    proj_dim: int = 64
    proj_max_batch_size: int = 64
    proj_seed: int = 0


class LoGraProjectionParams(GeneralProjectionParams):
    """Projection params for LoGra attributor."""


class FactGrassProjectionParams(GeneralProjectionParams):
    """Projection params for FactGraSS attributor."""


class TracInProjectionParams(GeneralProjectionParams):
    """Projection params for TracIn attributor."""


class DVEmbProjectionParams(ProjectionParams):
    """Projection params for DVEmb; proj_dim can be None for no projection."""

    proj_dim: Optional[int] = None


class RandomProjectionParams(ProjectionParams):
    """Params for random_project(); adds feature_batch_size and device."""

    feature_batch_size: int
    device: torch.device

    @field_validator("device", mode="before")
    @classmethod
    def _device_to_torch(cls, v: object) -> torch.device:
        if isinstance(v, torch.device):
            return v
        return torch.device(str(v))
