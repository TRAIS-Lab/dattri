"""Pydantic models for projection parameters."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import BaseModel, ConfigDict, field_validator


class GeneralProjectionParams(BaseModel):
    """General projection params used by LoGra, FactGraSS, TracIn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class LoGraProjectionParams(GeneralProjectionParams):
    """Projection params for LoGra attributor."""

    proj_dim: int = 4096


class FactGrassProjectionParams(GeneralProjectionParams):
    """Projection params for FactGraSS attributor."""

    proj_dim: int = 4096


class TracInProjectionParams(GeneralProjectionParams):
    """Projection params for TracIn attributor."""

    proj_dim: int = 512
    proj_max_batch_size: int = 32


class TRAKProjectionParams(GeneralProjectionParams):
    """Projection params for TRAK attributor."""

    proj_dim: int = 512
    proj_max_batch_size: int = 32


class DVEmbProjectionParams(GeneralProjectionParams):
    """Projection params for DVEmb; proj_dim can be None for no projection."""

    proj_dim: Optional[int] = None
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
