"""Async prefetching dataset for efficient chunk loading."""
from __future__ import annotations

import logging
import operator
import os
import pathlib
import queue
import threading
import time
from typing import Any, Dict, List

import torch
import torch.utils.data

logger = logging.getLogger(__name__)

# Forward declaration for lazy imports
ChunkedMemoryMapHandler = None


def _lazy_import_memory_map() -> None:
    """Lazy import of memory map module to avoid circular imports."""
    global ChunkedMemoryMapHandler
    if ChunkedMemoryMapHandler is None:
        try:
            from .memory_map import ChunkedMemoryMapHandler
        except ImportError:
            logger.warning("Failed to import ChunkedMemoryMapHandler")


class AsyncPrefetchDataset(torch.utils.data.Dataset):
    """Dataset that asynchronously prefetches chunks ahead of time."""

    def __init__(
        self,
        disk_io,
        data_type="gradients",
        batch_range=None,
        prefetch_factor=2,
    ) -> None:
        """Initialize async prefetching dataset.

        Args:
            disk_io: ChunkedDiskIOManager instance
            data_type: Type of data to load ("gradients" or "ifvp")
            batch_range: Optional tuple of (start_batch, end_batch) to filter batches
            prefetch_factor: Number of chunks to prefetch ahead
        """
        self.disk_io = disk_io
        self.data_type = data_type
        self.batch_range = batch_range
        self.prefetch_factor = prefetch_factor

        # Get chunk information
        self.chunk_info = self._load_chunk_info()

        # Prefetch queue and cache
        self._prefetch_queue = queue.Queue(maxsize=prefetch_factor * 2)
        self._chunk_cache = {}  # idx -> (tensor, batch_mapping)
        self._cache_lock = threading.Lock()

        # Prefetch thread
        self._prefetch_thread = None
        self._stop_prefetch = threading.Event()
        self._current_idx = 0
        self._idx_lock = threading.Lock()

        # Start prefetching
        self._start_prefetch_thread()

        logger.debug(
            f"AsyncPrefetchDataset: Found {len(self.chunk_info)} chunks for {data_type}",
        )

    def _load_chunk_info(self) -> List[Dict[str, Any]]:
        """Load information about all available chunks.

        Returns:
            List[Dict[str, Any]]: List of chunk information dictionaries containing metadata about each chunk.
        """
        chunk_info = []

        # Get subdirectory
        subdir = self.disk_io._get_chunk_subdir(self.data_type)
        chunk_path = os.path.join(self.disk_io.cache_dir, subdir)

        if not pathlib.Path(chunk_path).exists():
            return chunk_info

        # Find all chunk files
        _lazy_import_memory_map()
        chunk_files = ChunkedMemoryMapHandler.find_chunk_files(
            chunk_path,
            self.data_type,
        )

        for chunk_filename in chunk_files:
            try:
                # Load chunk metadata
                metadata = ChunkedMemoryMapHandler.read_chunk_metadata(
                    chunk_path,
                    chunk_filename,
                )

                # Check if chunk contains batches in our range
                if self.batch_range is not None:
                    start_batch, end_batch = self.batch_range
                    chunk_batches = [b["batch_idx"] for b in metadata["batches"]]

                    # Skip chunks that don't overlap with our range
                    if not any(start_batch <= idx < end_batch for idx in chunk_batches):
                        continue

                chunk_info.append(
                    {
                        "chunk_filename": chunk_filename,
                        "chunk_path": chunk_path,
                        "metadata": metadata,
                    },
                )

            except Exception as e:
                logger.warning("Error loading chunk %s: %s", chunk_filename, e)
                continue

        # Sort by chunk filename for consistency
        chunk_info.sort(key=operator.itemgetter("chunk_filename"))
        return chunk_info

    def _start_prefetch_thread(self) -> None:
        """Start the prefetch thread."""
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
        )
        self._prefetch_thread.start()

    def _prefetch_worker(self) -> None:
        """Worker thread that prefetches chunks."""
        _lazy_import_memory_map()

        while not self._stop_prefetch.is_set():
            try:
                # Get current index
                with self._idx_lock:
                    current_idx = self._current_idx

                # Determine which chunks to prefetch
                prefetch_indices = []
                for i in range(self.prefetch_factor):
                    idx = current_idx + i
                    if idx < len(self.chunk_info):
                        with self._cache_lock:
                            if idx not in self._chunk_cache:
                                prefetch_indices.append(idx)

                # Prefetch chunks
                for idx in prefetch_indices:
                    if self._stop_prefetch.is_set():
                        break

                    try:
                        # Load chunk
                        chunk_info = self.chunk_info[idx]
                        chunk_filename = chunk_info["chunk_filename"]
                        chunk_path = chunk_info["chunk_path"]

                        start_time = time.time()

                        # Load chunk as tensor with batch mapping
                        tensor, batch_mapping = (
                            ChunkedMemoryMapHandler.load_chunk_batch_range(
                                chunk_path,
                                chunk_filename,
                                self.batch_range,
                            )
                        )

                        load_time = time.time() - start_time

                        # Add to cache
                        with self._cache_lock:
                            self._chunk_cache[idx] = (tensor, batch_mapping)

                            # Evict old entries if cache is too large
                            max_cache_size = self.prefetch_factor * 3
                            if len(self._chunk_cache) > max_cache_size:
                                # Find and remove chunks that are far from current index
                                for cached_idx in list(self._chunk_cache.keys()):
                                    if cached_idx < current_idx - 1:
                                        del self._chunk_cache[cached_idx]

                        logger.debug(f"Prefetched chunk {idx} in {load_time:.3f}s")

                    except Exception as e:
                        logger.exception("Error prefetching chunk %s: %s", idx, e)

                # Small sleep to avoid busy waiting
                time.sleep(0.01)

            except Exception as e:
                logger.exception("Prefetch worker error: %s", e)
                time.sleep(0.1)

    def __len__(self) -> int:
        return len(self.chunk_info)

    def __getitem__(self, idx):
        """Get a chunk, using prefetched data when available.

        Args:
            idx: Index of the chunk to retrieve.

        Returns:
            Tuple of (tensor, batch_mapping) where:
            - tensor has shape (total_samples_in_chunk, total_proj_dim)
            - batch_mapping maps batch_idx to (start_row, end_row) in tensor

        Raises:
            IndexError: If idx is out of range for the available chunks.
        """
        if idx >= len(self.chunk_info):
            msg = f"Index {idx} out of range for {len(self.chunk_info)} chunks"
            raise IndexError(
                msg,
            )

        # Update current index for prefetcher
        with self._idx_lock:
            self._current_idx = idx

        # Check cache first
        with self._cache_lock:
            if idx in self._chunk_cache:
                return self._chunk_cache[idx]

        # If not in cache, load synchronously
        logger.debug("Cache miss for chunk %s, loading synchronously", idx)

        chunk_info = self.chunk_info[idx]
        chunk_filename = chunk_info["chunk_filename"]
        chunk_path = chunk_info["chunk_path"]

        _lazy_import_memory_map()
        try:
            # Load chunk as tensor with batch mapping
            tensor, batch_mapping = ChunkedMemoryMapHandler.load_chunk_batch_range(
                chunk_path,
                chunk_filename,
                self.batch_range,
            )

            # Add to cache for potential reuse
            with self._cache_lock:
                self._chunk_cache[idx] = (tensor, batch_mapping)

            return tensor, batch_mapping

        except Exception as e:
            logger.exception("Error loading chunk %s: %s", chunk_filename, e)
            # Return empty data on error
            layer_dims = chunk_info["metadata"].get("layer_dims", [])
            total_proj_dim = sum(layer_dims) if layer_dims else 0
            empty_tensor = torch.empty(0, total_proj_dim)
            return empty_tensor, {}

    def __del__(self) -> None:
        """Clean up prefetch thread."""
        if hasattr(self, "_stop_prefetch"):
            self._stop_prefetch.set()
        if hasattr(self, "_prefetch_thread") and self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)


def async_collate_fn(batch):
    """Collate function for async tensor chunks.

    Args:
        batch: List of (tensor, batch_mapping) tuples

    Returns:
        Combined tensor and mapping
    """
    if len(batch) == 1:
        return batch[0]

    # Combine multiple chunks
    tensors = []
    combined_mapping = {}
    row_offset = 0

    for tensor, batch_mapping in batch:
        tensors.append(tensor)

        # Update mapping with new offsets
        for batch_idx, (start, end) in batch_mapping.items():
            batch_size = end - start
            combined_mapping[batch_idx] = (
                row_offset + start,
                row_offset + start + batch_size,
            )

        row_offset += tensor.shape[0]

    # Concatenate tensors
    combined_tensor = torch.cat(tensors, dim=0)

    return combined_tensor, combined_mapping


class PrefetchDataLoader(torch.utils.data.DataLoader):
    """Custom DataLoader with GPU transfer overlap."""

    def __init__(self, dataset, device="cuda", transfer_stream=None, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.device = device
        self.transfer_stream = (
            transfer_stream or torch.cuda.Stream()
            if torch.cuda.is_available()
            else None
        )

    def __iter__(self):
        """Iterator with overlapped GPU transfers.

        Yields:
            Tuple of (tensor, batch_mapping) where tensor is moved to the target device
            and batch_mapping contains the batch index mappings.
        """
        base_iterator = super().__iter__()

        if self.transfer_stream is None or self.device == "cpu":
            # No overlap possible
            for data in base_iterator:
                yield data
            return

        # Prefetch first batch
        try:
            current_data = next(base_iterator)
            current_tensor, current_mapping = current_data

            # Start transferring first batch
            with torch.cuda.stream(self.transfer_stream):
                current_gpu = current_tensor.to(self.device, non_blocking=True)

        except StopIteration:
            return

        # Process remaining batches with overlap
        for next_data in base_iterator:
            next_tensor, next_mapping = next_data

            # Start transferring next batch while current is being used
            with torch.cuda.stream(self.transfer_stream):
                next_gpu = next_tensor.to(self.device, non_blocking=True)

            # Wait for current batch transfer to complete
            torch.cuda.current_stream().wait_stream(self.transfer_stream)

            # Yield current batch
            yield current_gpu, current_mapping

            # Swap batches
            current_gpu = next_gpu
            current_mapping = next_mapping

        # Yield last batch
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        yield current_gpu, current_mapping


def create_tensor_dataloader(
    disk_io,
    data_type="gradients",
    batch_size=4,
    pin_memory=True,
    batch_range=None,
    prefetch_factor=4,
    device="cuda",
) -> torch.utils.data.DataLoader:
    """Create an optimized async DataLoader for tensor-based chunked data.

    Args:
        disk_io: ChunkedDiskIOManager instance
        data_type: Type of data to load ("gradients" or "ifvp")
        batch_size: Number of chunks to load at once
        pin_memory: Whether to pin memory
        batch_range: Optional range of batches to include
        prefetch_factor: Number of chunks to prefetch ahead
        device: Target device for GPU transfer overlap

    Returns:
        DataLoader for efficient loading of chunked tensor data
    """
    dataset = AsyncPrefetchDataset(
        disk_io=disk_io,
        data_type=data_type,
        batch_range=batch_range,
        prefetch_factor=prefetch_factor,
    )

    # Use custom DataLoader with GPU transfer overlap
    return PrefetchDataLoader(
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=0,  # Prefetching is handled by the dataset
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=async_collate_fn,
        persistent_workers=False,
    )
