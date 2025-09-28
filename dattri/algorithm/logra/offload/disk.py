"""Enhanced disk offload strategy with async pipeline."""
from __future__ import annotations

import logging
import os
import pathlib
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from .disk_io.manager import ChunkedDiskIOManager
from .offload import Offload

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DiskOffloadManager(Offload):
    """Enhanced strategy that stores data on disk using async pipeline with buffer pooling."""

    def __init__(
        self,
        device: str,
        layer_names: List[str],
        cache_dir: Optional[str] = None,
        chunk_size: int = 32,
    ) -> None:
        if cache_dir is None:
            msg = "Cache directory must be provided for disk offload"
            raise ValueError(msg)

        self.device = device
        self.layer_names = layer_names
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size

        self.disk_io = ChunkedDiskIOManager(
            cache_dir,
            "default",
            hessian="raw",
            chunk_size=chunk_size,
            buffer_pool_size=4,
            write_queue_size=8,
        )

        # Track current batch range being processed
        self.current_batch_range = None

    def start_batch_range_processing(self, start_batch: int, end_batch: int) -> None:
        """Start processing a new batch range.

        Args:
            start_batch: Starting batch index for the range.
            end_batch: Ending batch index for the range.
        """
        self.current_batch_range = (start_batch, end_batch)
        self.disk_io.start_batch_range(start_batch, end_batch)

    def finish_batch_range_processing(self) -> None:
        """Finish processing the current batch range and write chunks."""
        if self.current_batch_range is not None:
            self.disk_io.finalize_batch_range()
            self.current_batch_range = None

    def store_gradients(
        self,
        batch_idx: int,
        gradients: List[torch.Tensor],
        is_test: bool = False,
    ) -> None:
        """Store gradients for a batch on disk using async pipeline.

        Args:
            batch_idx: Index of the batch to store gradients for.
            gradients: List of gradient tensors to store.
            is_test: Whether these are test gradients or training gradients.
        """
        self.disk_io.store_gradients(batch_idx, gradients, is_test)

    def retrieve_gradients(
        self,
        batch_idx: int,
        is_test: bool = False,
    ) -> List[torch.Tensor]:
        """Retrieve gradients for a batch from disk and move to device.

        Args:
            batch_idx: Index of the batch to retrieve gradients for.
            is_test: Whether to retrieve test gradients or training gradients.

        Returns:
            List[torch.Tensor]: List of gradient tensors moved to the compute device.
        """
        gradients = self.disk_io.retrieve_gradients(batch_idx, is_test)
        result = []
        for grad in gradients:
            if grad.numel() > 0:
                result.append(grad.to(self.device))
            else:
                result.append(torch.tensor([], device=self.device))
        return result

    def store_preconditioner(
        self,
        layer_idx: int,
        preconditioner: torch.Tensor,
    ) -> None:
        """Store a preconditioner for a layer on disk.

        Args:
            layer_idx: Index of the layer to store preconditioner for.
            preconditioner: Preconditioner tensor to store.
        """
        self.disk_io.store_preconditioner(layer_idx, preconditioner)

    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Retrieve a preconditioner for a layer from disk and move to device.

        Args:
            layer_idx: Index of the layer to retrieve preconditioner for.

        Returns:
            Optional[torch.Tensor]: Preconditioner tensor moved to compute device, or None if not found.
        """
        preconditioner = self.disk_io.retrieve_preconditioner(layer_idx)
        if preconditioner is not None:
            return preconditioner.to(self.device)
        return None

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """Store IFVP for a batch on disk using async pipeline.

        Args:
            batch_idx: Index of the batch to store IFVP for.
            ifvp: List of IFVP tensors to store.
        """
        self.disk_io.store_ifvp(batch_idx, ifvp)

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """Retrieve IFVP for a batch from disk and move to device.

        Args:
            batch_idx: Index of the batch to retrieve IFVP for.

        Returns:
            List[torch.Tensor]: List of IFVP tensors moved to the compute device.
        """
        ifvp_list = self.disk_io.retrieve_ifvp(batch_idx)
        result = []
        for ifvp in ifvp_list:
            if ifvp.numel() > 0:
                result.append(ifvp.to(self.device))
            else:
                result.append(torch.tensor([], device=self.device))
        return result

    def create_gradient_dataloader(
        self,
        data_type: str,
        batch_size: int = 1,
        pin_memory: bool = True,
        batch_range: Optional[Tuple[int, int]] = None,
        is_test: bool = False,
    ) -> DataLoader:
        """Create an optimized DataLoader with async prefetching.

        Args:
            data_type: Type of data to load
            batch_size: Number of chunks to load at once
            pin_memory: Whether to pin memory
            batch_range: Optional batch range filter
            is_test: Whether loading test data (unused)

        Returns:
            DataLoader instance with async prefetching
        """
        return self.disk_io.create_gradient_dataloader(
            data_type=data_type,
            batch_size=batch_size,
            pin_memory=pin_memory,
            batch_range=batch_range,
        )

    def has_preconditioners(self) -> bool:
        """Check if preconditioners are available on disk.

        Returns:
            bool: True if preconditioners are available, False otherwise.
        """
        return self.disk_io.has_preconditioners()

    def has_ifvp(self) -> bool:
        """Check if IFVP are available on disk.

        Returns:
            bool: True if IFVP data is available, False otherwise.
        """
        return self.disk_io.has_ifvp()

    def clear_cache(self) -> None:
        """Clear all cached data from disk."""
        self.disk_io.wait_for_async_operations()

        for subdir in ["grad", "ifvp", "precond"]:
            subdir_path = os.path.join(self.cache_dir, subdir)
            if pathlib.Path(subdir_path).exists():
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    try:
                        pathlib.Path(file_path).unlink()
                    except Exception as e:
                        logger.warning("Error removing %s: %s", file_path, e)

    def wait_for_async_operations(self) -> None:
        """Wait for any pending asynchronous disk operations to complete."""
        self.disk_io.wait_for_async_operations()

    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the compute device.

        Args:
            tensor: Tensor to move to the compute device.

        Returns:
            torch.Tensor: Tensor moved to the compute device.
        """
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def move_from_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor from the compute device to CPU for disk storage.

        Args:
            tensor: Tensor to move from the compute device to CPU.

        Returns:
            torch.Tensor: Tensor moved to CPU for disk storage.
        """
        return tensor.cpu() if tensor.device.type != "cpu" else tensor
