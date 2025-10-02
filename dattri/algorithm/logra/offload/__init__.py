"""Strategies for different memory management approaches."""
from __future__ import annotations

from typing import List, Literal, Optional

from .cpu import CPUOffloadManager
from .disk import DiskOffloadManager
from .memory import MemoryOffloadManager

# Type definitions
OffloadOptions = Literal["none", "cpu", "disk"]


def create_offload_manager(
    offload_type: OffloadOptions,
    device: str,
    layer_names: List[str],
    cache_dir: Optional[str] = None,
    chunk_size: int = 32,
):
    """Factory function to create appropriate offload strategy.

    Args:
        offload_type: Type of offload strategy
        device: Compute device
        layer_names: Names of layers
        cache_dir: Cache directory (required for disk offload)
        chunk_size: Chunk size for disk offload

    Returns:
        Appropriate offload strategy instance

    Raises:
        ValueError: If offload_type is not one of "none", "cpu", or "disk".
    """
    if offload_type == "none":
        return MemoryOffloadManager(device, layer_names, cache_dir)
    if offload_type == "cpu":
        return CPUOffloadManager(device, layer_names, cache_dir)
    if offload_type == "disk":
        return DiskOffloadManager(device, layer_names, cache_dir, chunk_size)
    msg = f"Unknown offload type: {offload_type}"
    raise ValueError(msg)


__all__ = [
    "CPUOffloadManager",
    "DiskOffloadManager",
    "MemoryOffloadManager",
    "OffloadOptions",
    "create_offload_manager",
]
