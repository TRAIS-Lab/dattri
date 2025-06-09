from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

class Offload(ABC):
    """
    Abstract base class defining the interface for offload strategies.
    Each strategy handles a specific method of data storage and retrieval.
    """

    @abstractmethod
    def __init__(self, device: str, layer_names: List[str], cache_dir: Optional[str] = None):
        """
        Initialize the offload strategy.

        Args:
            device: The primary compute device
            layer_names: Names of layers being analyzed
            cache_dir: Directory for caching data (may be None for non-disk strategies)
        """
        pass

    @abstractmethod
    def store_gradients(self, batch_idx: int, gradients: List[torch.Tensor], is_test: bool = False) -> None:
        """
        Store gradients for a batch.

        Args:
            batch_idx: Batch index
            gradients: List of gradient tensors (one per layer)
            is_test: Whether these are test gradients
        """
        pass

    @abstractmethod
    def retrieve_gradients(self, batch_idx: int, is_test: bool = False) -> List[torch.Tensor]:
        """
        Retrieve gradients for a batch.

        Args:
            batch_idx: Batch index
            is_test: Whether to retrieve test gradients

        Returns:
            List of gradient tensors (one per layer)
        """
        pass

    @abstractmethod
    def store_preconditioner(self, layer_idx: int, preconditioner: torch.Tensor) -> None:
        """
        Store a preconditioner for a layer.

        Args:
            layer_idx: Layer index
            preconditioner: Preconditioner tensor
        """
        pass

    @abstractmethod
    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve a preconditioner for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Preconditioner tensor, or None if not found
        """
        pass

    @abstractmethod
    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """
        Store IFVP (inverse-Hessian-vector product) for a batch.

        Args:
            batch_idx: Batch index
            ifvp: List of IFVP tensors (one per layer)
        """
        pass

    @abstractmethod
    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """
        Retrieve IFVP for a batch.

        Args:
            batch_idx: Batch index

        Returns:
            List of IFVP tensors (one per layer)
        """
        pass

    @abstractmethod
    def create_gradient_dataloader(
            self,
            data_type: str,
            batch_size: int = 1,
            num_workers: int = 4,
            pin_memory: bool = True,
            batch_range: Optional[Tuple[int, int]] = None,
            is_test: bool = False
        ) -> Optional[DataLoader]:
        """
        Create a DataLoader for loading data (if applicable).

        Args:
            data_type: Type of data to load ("gradients" or "ifvp")
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            batch_range: Optional range of batches to include
            is_test: Whether to load test data

        Returns:
            DataLoader or None if not applicable
        """
        pass

    @abstractmethod
    def has_preconditioners(self) -> bool:
        """
        Check if preconditioners are available.

        Returns:
            True if preconditioners are available, False otherwise
        """
        pass

    @abstractmethod
    def has_ifvp(self) -> bool:
        """
        Check if IFVP are available.

        Returns:
            True if any IFVP are available, False otherwise
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        pass

    @abstractmethod
    def wait_for_async_operations(self) -> None:
        """
        Wait for any pending asynchronous operations to complete.
        """
        pass

    @abstractmethod
    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move a tensor to the compute device.

        Args:
            tensor: Input tensor

        Returns:
            Tensor on the compute device
        """
        pass

    @abstractmethod
    def move_from_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move a tensor from the compute device to storage.

        Args:
            tensor: Input tensor on compute device

        Returns:
            Tensor in storage format
        """
        pass