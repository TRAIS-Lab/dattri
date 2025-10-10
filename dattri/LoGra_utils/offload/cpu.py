"""
CPU offload strategy.
"""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .offload import Offload

import logging
logger = logging.getLogger(__name__)

class CPUOffloadManager(Offload):
    """
    Strategy that stores data on CPU and moves to device when needed.
    """

    def __init__(self, device: str, layer_names: List[str], cache_dir: Optional[str] = None):
        """
        Initialize the CPU offload strategy.

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

        # Store as concatenated tensors on CPU for efficiency
        self.cached_gradients = {}  # batch_idx -> tensor (CPU)
        self.cached_test_gradients = {}  # batch_idx -> tensor (CPU)
        self.preconditioners = [None] * len(layer_names)
        self.cached_ifvp = {}  # batch_idx -> tensor (CPU)

    def _ensure_dims_set(self, gradients: List[torch.Tensor]):
        """Ensure layer dimensions are set from first gradient batch."""
        if self.layer_dims is None:
            self.layer_dims = [g.shape[1] if g.numel() > 0 else 0 for g in gradients]
            self.total_proj_dim = sum(self.layer_dims)
            logger.debug(f"Detected layer dimensions: {len(self.layer_dims)} layers, total={self.total_proj_dim}")

    def _concatenate_gradients(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate list of gradient tensors into single tensor on CPU."""
        self._ensure_dims_set(gradients)

        batch_size = next((g.shape[0] for g in gradients if g.numel() > 0), 0)
        if batch_size == 0:
            return torch.empty(0, self.total_proj_dim, device='cpu')

        # Pre-allocate result tensor on CPU
        result = torch.zeros(batch_size, self.total_proj_dim, device='cpu')

        # Fill in each layer's data
        start_idx = 0
        for layer_idx, (grad, dim) in enumerate(zip(gradients, self.layer_dims)):
            end_idx = start_idx + dim
            if grad.numel() > 0:
                result[:, start_idx:end_idx] = grad.cpu() if grad.device.type != 'cpu' else grad
            start_idx = end_idx

        return result

    def _split_tensor(self, tensor: torch.Tensor, to_device: bool = True) -> List[torch.Tensor]:
        """Split concatenated tensor back into per-layer tensors."""
        if self.layer_dims is None:
            raise ValueError("Layer dimensions not set")

        result = []
        start_idx = 0
        for dim in self.layer_dims:
            end_idx = start_idx + dim
            layer_tensor = tensor[:, start_idx:end_idx].contiguous()
            if to_device and self.device != 'cpu':
                layer_tensor = layer_tensor.to(self.device)
            result.append(layer_tensor)
            start_idx = end_idx

        return result

    def store_gradients(self, batch_idx: int, gradients: List[torch.Tensor], is_test: bool = False) -> None:
        """
        Store gradients for a batch on CPU as concatenated tensor.

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

    def retrieve_gradients(self, batch_idx: int, is_test: bool = False) -> List[torch.Tensor]:
        """
        Retrieve gradients for a batch and move to device.

        Args:
            batch_idx: Batch index
            is_test: Whether to retrieve test gradients

        Returns:
            List of gradient tensors (one per layer) on the compute device
        """
        cached_dict = self.cached_test_gradients if is_test else self.cached_gradients

        if batch_idx not in cached_dict:
            if self.layer_dims is None:
                return [torch.tensor([], device=self.device) for _ in self.layer_names]
            else:
                return [torch.zeros(0, dim, device=self.device) for dim in self.layer_dims]

        # Split and move to device
        return self._split_tensor(cached_dict[batch_idx], to_device=True)

    def store_preconditioner(self, layer_idx: int, preconditioner: Optional[torch.Tensor]) -> None:
        """
        Store a preconditioner for a layer on CPU.

        Args:
            layer_idx: Layer index
            preconditioner: Preconditioner tensor (can be None)
        """
        if layer_idx < len(self.preconditioners):
            if preconditioner is not None:
                self.preconditioners[layer_idx] = preconditioner.cpu() if preconditioner.device.type != 'cpu' else preconditioner
            else:
                self.preconditioners[layer_idx] = None

    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve a preconditioner for a layer and move to device.

        Args:
            layer_idx: Layer index

        Returns:
            Preconditioner tensor on the compute device, or None if not found
        """
        if layer_idx >= len(self.preconditioners) or self.preconditioners[layer_idx] is None:
            return None

        return self.preconditioners[layer_idx].to(self.device)

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """
        Store IFVP for a batch on CPU as concatenated tensor.

        Args:
            batch_idx: Batch index
            ifvp: List of IFVP tensors (one per layer)
        """
        concatenated = self._concatenate_gradients(ifvp)  # Same format as gradients
        self.cached_ifvp[batch_idx] = concatenated

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """
        Retrieve IFVP for a batch and move to device.

        Args:
            batch_idx: Batch index

        Returns:
            List of IFVP tensors (one per layer) on the compute device
        """
        if batch_idx not in self.cached_ifvp:
            if self.layer_dims is None:
                return [torch.tensor([], device=self.device) for _ in self.layer_names]
            else:
                return [torch.zeros(0, dim, device=self.device) for dim in self.layer_dims]

        # Split and move to device
        return self._split_tensor(self.cached_ifvp[batch_idx], to_device=True)

    def create_gradient_dataloader(
            self,
            data_type: str,
            batch_size: int = 1,
            pin_memory: bool = True,
            batch_range: Optional[Tuple[int, int]] = None,
            is_test: bool = False
        ) -> Optional[DataLoader]:
        """
        Create a DataLoader that returns CPU tensors and moves them to device.

        Args:
            data_type: Type of data to load
            batch_size: Number of batches to return at once
            pin_memory: Whether to pin memory
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
            batch_indices = [idx for idx in batch_indices if start_batch <= idx < end_batch]

        # Create simple dataset
        class CPUTensorDataset(torch.utils.data.Dataset):
            def __init__(self, cache, batch_indices):
                self.cache = cache
                self.batch_indices = batch_indices

            def __len__(self):
                return len(self.batch_indices)

            def __getitem__(self, idx):
                batch_idx = self.batch_indices[idx]
                cpu_tensor = self.cache[batch_idx]
                # Keep on CPU - DataLoader will handle pinning if requested
                # Return in same format as disk loader
                batch_mapping = {batch_idx: (0, cpu_tensor.shape[0])}
                return cpu_tensor, batch_mapping

        dataset = CPUTensorDataset(cache, batch_indices)

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
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

    def has_preconditioners(self) -> bool:
        """
        Check if preconditioners are available.

        Returns:
            True if any preconditioners are available, False otherwise
        """
        return any(p is not None for p in self.preconditioners)

    def has_ifvp(self) -> bool:
        """
        Check if IFVP are available.

        Returns:
            True if any IFVP are available, False otherwise
        """
        return len(self.cached_ifvp) > 0

    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        self.cached_gradients = {}
        self.cached_test_gradients = {}
        self.preconditioners = [None] * len(self.layer_names)
        self.cached_ifvp = {}
        # Don't clear layer_dims as they might be needed later

    def wait_for_async_operations(self) -> None:
        """
        No asynchronous operations for CPU offload.
        """
        pass

    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move a tensor to the compute device.

        Args:
            tensor: Input tensor (possibly on CPU)

        Returns:
            Tensor on the compute device
        """
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def move_from_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move a tensor from the compute device to CPU.

        Args:
            tensor: Input tensor on compute device

        Returns:
            Tensor on CPU
        """
        return tensor.cpu() if tensor.device.type != 'cpu' else tensor