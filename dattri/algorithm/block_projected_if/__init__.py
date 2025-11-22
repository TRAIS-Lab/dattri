"""BlockProjectedIF utilities package.

Contains all dependencies for block-projected influence function attributor.
"""

from .block_projected_if import BlockProjectedIFAttributor
from .core.compressor import (
    Compressor,
    setup_model_compressors,
)
from .core.hook import HookManager
from .core.metadata import MetadataManager
from .core.utils import stable_inverse
from .offload import create_offload_manager

__all__ = [
    "BlockProjectedIFAttributor",
    "Compressor",
    "HookManager",
    "MetadataManager",
    "create_offload_manager",
    "setup_model_compressors",
    "stable_inverse",
]
