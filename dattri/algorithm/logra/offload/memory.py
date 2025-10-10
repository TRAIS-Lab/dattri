"""Memory offload strategy."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from .offload import Offload

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MemoryOffloadManager(Offload):
    """Strategy that keeps all data in memory on the specified device."""

    def __init__(
        self,
        device: str,
        layer_names: List[str],
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize the memory offload strategy.

        Args:
            device: The primary compute device
            layer_names: Names of layers being analyzed
            cache_dir: Directory for caching data (ignored for this strategy)
        """
        self.device = device
        self.layer_names = layer_names
        self.layer_dims = None
        self.total_proj_dim = None
        self.cache_dir = cache_dir  # Keep for consistency

        # Store as concatenated tensors for efficiency
        self.cached_gradients = {}  # batch_idx -> tensor
        self.cached_test_gradients = {}  # batch_idx -> tensor
        self.preconditioners = [None] * len(layer_names)
        self.cached_ifvp = {}  # batch_idx -> tensor

    def _ensure_dims_set(self, gradients: List[torch.Tensor]) -> None:
        """Ensure layer dimensions are set from first gradient batch.

        Args:
            gradients: List of gradient tensors to extract dimensions from.
        """
        if self.layer_dims is None:
            self.layer_dims = [g.shape[1] if g.numel() > 0 else 0 for g in gradients]
            self.total_proj_dim = sum(self.layer_dims)
            logger.debug(
                f"Detected layer dimensions: {len(self.layer_dims)} layers, total={self.total_proj_dim}",
            )

    def _concatenate_gradients(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate list of gradient tensors into single tensor.

        Args:
            gradients: List of gradient tensors to concatenate.

        Returns:
            torch.Tensor: Concatenated tensor with shape (batch_size, total_proj_dim).
        """
        self._ensure_dims_set(gradients)

        batch_size = next((g.shape[0] for g in gradients if g.numel() > 0), 0)
        if batch_size == 0:
            return torch.empty(0, self.total_proj_dim, device=self.device)

        # Pre-allocate result tensor
        result = torch.zeros(batch_size, self.total_proj_dim, device=self.device)

        # Fill in each layer's data
        start_idx = 0
        for _layer_idx, (grad, dim) in enumerate(zip(gradients, self.layer_dims)):
            end_idx = start_idx + dim
            if grad.numel() > 0:
                result[:, start_idx:end_idx] = grad.to(self.device)
            start_idx = end_idx

        return result

    def _split_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Split concatenated tensor back into per-layer tensors.

        Args:
            tensor: Concatenated tensor to split.

        Returns:
            List[torch.Tensor]: List of per-layer tensors.

        Raises:
            ValueError: If layer dimensions are not set.
        """
        if self.layer_dims is None:
            msg = "Layer dimensions not set"
            raise ValueError(msg)

        result = []
        start_idx = 0
        for dim in self.layer_dims:
            end_idx = start_idx + dim
            result.append(tensor[:, start_idx:end_idx].contiguous())
            start_idx = end_idx

        return result

    def store_gradients(
        self,
        batch_idx: int,
        gradients: List[torch.Tensor],
        is_test: bool = False,
    ) -> None:
        """Store gradients for a batch in memory as concatenated tensor.

        Args:
            batch_idx: Batch index
            gradients: List of gradient tensors (one per layer)
            is_test: Whether these are test gradients
        """
        concatenated = self._concatenate_gradients(gradients)

        if is_test:
            self.cached_test_gradients[batch_idx] = concatenated
        else:
            self.cached_gradients[batch_idx] = concatenated

    def retrieve_gradients(
        self,
        batch_idx: int,
        is_test: bool = False,
    ) -> List[torch.Tensor]:
        """Retrieve gradients for a batch from memory.

        Args:
            batch_idx: Batch index
            is_test: Whether to retrieve test gradients

        Returns:
            List of gradient tensors (one per layer)
        """
        if is_test:
            if batch_idx not in self.cached_test_gradients:
                if self.layer_dims is None:
                    return [
                        torch.tensor([], device=self.device) for _ in self.layer_names
                    ]
                return [
                    torch.zeros(0, dim, device=self.device) for dim in self.layer_dims
                ]
            return self._split_tensor(self.cached_test_gradients[batch_idx])
        if batch_idx not in self.cached_gradients:
            if self.layer_dims is None:
                return [torch.tensor([], device=self.device) for _ in self.layer_names]
            return [torch.zeros(0, dim, device=self.device) for dim in self.layer_dims]
        return self._split_tensor(self.cached_gradients[batch_idx])

    def store_preconditioner(
        self,
        layer_idx: int,
        preconditioner: Optional[torch.Tensor],
    ) -> None:
        """Store a preconditioner for a layer in memory.

        Args:
            layer_idx: Layer index
            preconditioner: Preconditioner tensor (can be None)
        """
        if layer_idx < len(self.preconditioners):
            self.preconditioners[layer_idx] = preconditioner

    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Retrieve a preconditioner for a layer from memory.

        Args:
            layer_idx: Layer index

        Returns:
            Preconditioner tensor, or None if not found
        """
        if layer_idx >= len(self.preconditioners):
            return None
        return self.preconditioners[layer_idx]

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """Store IFVP for a batch in memory as concatenated tensor.

        Args:
            batch_idx: Batch index
            ifvp: List of IFVP tensors (one per layer)
        """
        concatenated = self._concatenate_gradients(ifvp)  # Same format as gradients
        self.cached_ifvp[batch_idx] = concatenated

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """Retrieve IFVP for a batch from memory.

        Args:
            batch_idx: Batch index

        Returns:
            List of IFVP tensors (one per layer)
        """
        if batch_idx not in self.cached_ifvp:
            if self.layer_dims is None:
                return [torch.tensor([], device=self.device) for _ in self.layer_names]
            return [torch.zeros(0, dim, device=self.device) for dim in self.layer_dims]
        return self._split_tensor(self.cached_ifvp[batch_idx])

    def create_gradient_dataloader(
        self,
        data_type: str,
        batch_size: int = 1,
        pin_memory: bool = True,
        batch_range: Optional[Tuple[int, int]] = None,
        is_test: bool = False,
    ) -> Optional[DataLoader]:
        """Create a simple DataLoader that returns tensors from memory.

        Args:
            data_type: Type of data to load
            batch_size: Number of batches to return at once
            pin_memory: Whether to pin memory (ignored)
            batch_range: Optional batch range filter
            is_test: Whether loading test data

        Returns:
            DataLoader-like iterator
        """
        # Select appropriate cache
        if data_type == "gradients":
            cache = self.cached_test_gradients if is_test else self.cached_gradients
        elif data_type == "ifvp":
            cache = self.cached_ifvp
        else:
            return None

        # Filter by batch range if specified
        batch_indices = sorted(cache.keys())
        if batch_range is not None:
            start_batch, end_batch = batch_range
            batch_indices = [
                idx for idx in batch_indices if start_batch <= idx < end_batch
            ]

        # Create simple dataset
        class MemoryTensorDataset(torch.utils.data.Dataset):
            def __init__(self, cache, batch_indices) -> None:
                self.cache = cache
                self.batch_indices = batch_indices

            def __len__(self) -> int:
                return len(self.batch_indices)

            def __getitem__(self, idx):
                batch_idx = self.batch_indices[idx]
                tensor = self.cache[batch_idx]
                # Return in same format as disk loader
                batch_mapping = {batch_idx: (0, tensor.shape[0])}
                return tensor, batch_mapping

        dataset = MemoryTensorDataset(cache, batch_indices)

        # Custom collate function to combine multiple batches
        def collate_fn(items):
            if len(items) == 1:
                return items[0]

            tensors = []
            combined_mapping = {}
            row_offset = 0

            for tensor, mapping in items:
                tensors.append(tensor)
                for batch_idx, (start, end) in mapping.items():
                    batch_size = end - start
                    combined_mapping[batch_idx] = (row_offset, row_offset + batch_size)
                row_offset += tensor.shape[0]

            combined_tensor = torch.cat(tensors, dim=0)
            return combined_tensor, combined_mapping

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def has_preconditioners(self) -> bool:
        """Check if preconditioners are available in memory.

        Returns:
            True if any preconditioners are available, False otherwise
        """
        return any(p is not None for p in self.preconditioners)

    def has_ifvp(self) -> bool:
        """Check if IFVP are available in memory.

        Returns:
            True if any IFVP are available, False otherwise
        """
        return len(self.cached_ifvp) > 0

    def clear_cache(self) -> None:
        """Clear all cached data from memory."""
        self.cached_gradients = {}
        self.cached_test_gradients = {}
        self.preconditioners = [None] * len(self.layer_names)
        self.cached_ifvp = {}
        # Don't clear layer_dims as they might be needed later

    def wait_for_async_operations(self) -> None:
        """No asynchronous operations for memory offload."""

    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the compute device.

        Args:
            tensor: Input tensor

        Returns:
            Tensor on the device
        """
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def move_from_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """No movement needed as we keep tensors on the device.

        Args:
            tensor: Input tensor

        Returns:
            Same tensor (stays on the device)
        """
        return tensor
