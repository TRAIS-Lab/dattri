"""Enhanced Disk I/O manager with buffer pooling and async pipeline."""

import logging
import os
import pathlib
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from .memory_map import ChunkedMemoryMapHandler

logger = logging.getLogger(__name__)

DataTypeOptions = Literal["gradients", "preconditioners", "ifvp"]
HessianOptions = Literal["none", "raw", "kfac", "ekfac"]


@dataclass
class ChunkBuffer:
    """Buffer for accumulating data in tensor format."""

    tensor: torch.Tensor  # Pre-allocated tensor
    batch_indices: List[int]  # Batch indices in this chunk
    batch_info: List[Dict[str, Any]]  # Batch metadata (batch_idx, start_row, end_row)
    current_row: int  # Next row to write to
    buffer_id: int  # Unique buffer ID for tracking


class BufferPool:
    """Pool of reusable buffers to avoid allocation overhead."""

    def __init__(
        self,
        pool_size: int,
        buffer_shape: Tuple[int, int],
        dtype=torch.float32,
    ) -> None:
        self.pool_size = pool_size
        self.buffer_shape = buffer_shape
        self.dtype = dtype
        self.available = queue.Queue(maxsize=pool_size)
        self.buffer_counter = 0
        self.total_created = 0
        self.wait_count = 0

        # Pre-allocate all buffers
        for _ in range(pool_size):
            buffer = ChunkBuffer(
                tensor=torch.zeros(buffer_shape, dtype=dtype, pin_memory=True),
                batch_indices=[],
                batch_info=[],
                current_row=0,
                buffer_id=self.buffer_counter,
            )
            self.buffer_counter += 1
            self.total_created += 1
            self.available.put(buffer)

        logger.info(
            "BufferPool initialized with %s buffers of shape %s",
            pool_size,
            buffer_shape,
        )

    def get_buffer(self, timeout: Optional[float] = None) -> Optional[ChunkBuffer]:
        """Get a buffer from the pool.

        Args:
            timeout: Maximum time to wait for a buffer in seconds. If None, waits indefinitely.

        Returns:
            Optional[ChunkBuffer]: A buffer from the pool, or None if timeout occurred.
        """
        try:
            # Log if we have to wait
            if self.available.empty():
                self.wait_count += 1
                if self.wait_count % 10 == 1:  # Log every 10th wait
                    logger.warning(
                        f"BufferPool: Waiting for buffers (wait count: {self.wait_count})",
                    )

            buffer = self.available.get(timeout=timeout)
            # Reset the buffer
            buffer.batch_indices.clear()
            buffer.batch_info.clear()
            buffer.current_row = 0
            return buffer
        except queue.Empty:
            logger.warning(
                f"BufferPool: Timeout getting buffer after {timeout}s. "
                f"Pool size: {self.pool_size}, total created: {self.total_created}",
            )
            return None

    def return_buffer(self, buffer: ChunkBuffer) -> None:
        """Return a buffer to the pool.

        Args:
            buffer: The buffer to return to the pool.
        """
        try:
            self.available.put_nowait(buffer)
        except queue.Full:
            logger.warning(
                f"Buffer pool full, discarding buffer {buffer.buffer_id}. "
                f"This should not happen - possible double return?",
            )


class ChunkedDiskIOManager:
    """Enhanced disk I/O manager with buffer pooling and async pipeline."""

    def __init__(
        self,
        cache_dir: str,
        setting: str,
        num_threads: int = 32,
        hessian: HessianOptions = "raw",
        chunk_size: int = 32,
        max_samples_per_chunk: int = 2048,
        buffer_pool_size: int = 8,
        write_queue_size: int = 32,
        num_write_workers: int = 4,
    ) -> None:
        self.cache_dir = cache_dir
        self.setting = setting
        self.num_threads = num_threads
        self.hessian = hessian
        self.chunk_size = chunk_size
        self.max_samples_per_chunk = max_samples_per_chunk
        self._shutdown = False

        # Create cache directory structure
        if cache_dir:
            pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)
            for subdir in ["grad", "ifvp", "precond"]:
                pathlib.Path(os.path.join(cache_dir, subdir)).mkdir(
                    exist_ok=True,
                    parents=True,
                )

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.futures = []

        # Buffer management
        self._chunk_buffers = {}  # (data_type, chunk_id) -> ChunkBuffer
        self._buffer_locks = defaultdict(threading.Lock)

        # Buffer pools for each data type
        self._buffer_pools = {}
        self.buffer_pool_size = buffer_pool_size

        # Write queue for async writes
        self._write_queue = queue.Queue(maxsize=write_queue_size)
        self._write_workers = []
        self._shutdown = False

        # Start multiple write workers
        for i in range(num_write_workers):
            worker = threading.Thread(
                target=self._write_worker,
                daemon=True,
                name=f"DiskWriter-{i}",
            )
            worker.start()
            self._write_workers.append(worker)

        logger.info("Started %s write workers", num_write_workers)

        # Pending writes tracking
        self._pending_writes = set()  # Track chunk_ids being written
        self._pending_writes_lock = threading.Lock()

        # Read cache for recently accessed chunks
        self._read_cache = (
            {}
        )  # (data_type, chunk_id) -> (tensor, metadata, last_access_time)
        self._read_cache_lock = threading.Lock()
        self._max_read_cache_size = 4

        # Track dimensions and batch range
        self.layer_dims = None
        self.total_proj_dim = None
        self.current_batch_range = None

        # Try to load layer dimensions from existing data
        self._load_layer_dims_from_metadata()

        logger.debug(
            "Initialized Enhanced ChunkedDiskIOManager with chunk_size=%s, buffer_pool_size=%s",
            chunk_size,
            buffer_pool_size,
        )

    def get_chunk_id(self, batch_idx: int) -> int:
        """Get chunk ID for a batch index.

        Args:
            batch_idx: Index of the batch.

        Returns:
            int: The chunk ID containing this batch.
        """
        return batch_idx // self.chunk_size

    def _load_layer_dims_from_metadata(self) -> None:
        """Try to load layer dimensions from existing chunk metadata."""
        if not self.cache_dir:
            return

        # Try gradient chunks first
        for data_type, subdir in [("gradients", "grad"), ("ifvp", "ifvp")]:
            data_dir = os.path.join(self.cache_dir, subdir)
            if pathlib.Path(data_dir).exists():
                try:
                    chunk_files = ChunkedMemoryMapHandler.find_chunk_files(
                        data_dir,
                        data_type,
                    )
                    if chunk_files:
                        metadata = ChunkedMemoryMapHandler.read_chunk_metadata(
                            data_dir,
                            chunk_files[0],
                        )
                        if "layer_dims" in metadata:
                            self.layer_dims = metadata["layer_dims"]
                            self.total_proj_dim = sum(self.layer_dims)
                            logger.info(
                                f"Loaded layer dimensions from {data_type} metadata: {self.layer_dims}",
                            )
                            return
                except Exception as e:
                    logger.debug("Could not load layer dims from %s: %s", data_type, e)

    def _ensure_layer_dims(self) -> None:
        """Ensure layer dimensions are available.

        Raises:
            ValueError: If layer dimensions are not available and cannot be loaded.
        """
        if self.layer_dims is None:
            self._load_layer_dims_from_metadata()

        if self.layer_dims is None:
            msg = (
                "Layer dimensions not available. Either compute gradients first or "
                "ensure existing gradient/IFVP chunks are present in the cache directory."
            )
            raise ValueError(
                msg,
            )

    def _get_or_create_buffer_pool(self, data_type: str) -> BufferPool:
        """Get or create a buffer pool for a data type.

        Args:
            data_type: Type of data ("gradients", "ifvp", etc.).

        Returns:
            BufferPool: Buffer pool for the specified data type.
        """
        if data_type not in self._buffer_pools:
            if self.total_proj_dim is None:
                self._ensure_layer_dims()

            # Use larger pool size to avoid exhaustion
            pool_size = max(8, self.buffer_pool_size)  # At least 8 buffers

            self._buffer_pools[data_type] = BufferPool(
                pool_size=pool_size,
                buffer_shape=(self.max_samples_per_chunk, self.total_proj_dim),
                dtype=torch.float32,
            )
            logger.info(
                "Created buffer pool for %s with %s buffers",
                data_type,
                pool_size,
            )

        return self._buffer_pools[data_type]

    def start_batch_range(self, start_batch: int, end_batch: int) -> None:
        """Start processing a batch range.

        Args:
            start_batch: Starting batch index for the range.
            end_batch: Ending batch index for the range.

        Raises:
            ValueError: If batch range start is not aligned to chunk_size.
        """
        if start_batch % self.chunk_size != 0:
            msg = f"Batch range start {start_batch} must be aligned to chunk_size {self.chunk_size}"
            raise ValueError(
                msg,
            )

        self.current_batch_range = (start_batch, end_batch)
        logger.info("Starting batch range [%s, %s)", start_batch, end_batch)

    def _write_worker(self) -> None:
        """Background worker thread for writing chunks to disk."""
        while not self._shutdown:
            try:
                # Get write task from queue
                write_task = self._write_queue.get(timeout=1.0)
                if write_task is None:  # Shutdown signal
                    break

                data_type, chunk_id, tensor, batch_info, buffer = write_task

                # Perform the write
                try:
                    self._write_chunk_tensor_sync(
                        data_type,
                        chunk_id,
                        tensor,
                        batch_info,
                    )

                    # Remove from pending writes
                    with self._pending_writes_lock:
                        self._pending_writes.discard((data_type, chunk_id))

                finally:
                    # Return buffer to pool if provided (for backward compatibility)
                    if buffer is not None:
                        pool = self._get_or_create_buffer_pool(data_type)
                        pool.return_buffer(buffer)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Write worker error: %s", e)
                import traceback

                traceback.print_exc()

    def _write_chunk_tensor_sync(
        self,
        data_type: str,
        chunk_id: int,
        tensor: torch.Tensor,
        batch_info: List[Dict[str, Any]],
    ) -> None:
        """Synchronously write a chunk tensor to disk.

        Args:
            data_type: Type of data being written.
            chunk_id: ID of the chunk being written.
            tensor: Tensor data to write.
            batch_info: List of batch metadata dictionaries.

        Raises:
            Exception: If there's an error writing the chunk to disk.
        """
        try:
            subdir = "grad" if data_type == "gradients" else data_type
            chunk_dir = os.path.join(self.cache_dir, subdir)

            # Write using memory map handler with async flush
            ChunkedMemoryMapHandler.write_chunk(
                chunk_dir,
                data_type,
                tensor,
                batch_info,
                self.layer_dims,
                dtype="float32",
                flush_mode="async",  # Use async mode to avoid blocking
            )

            logger.debug(f"Wrote {data_type} chunk {chunk_id}: shape={tensor.shape}")

        except Exception as e:
            logger.error("Error writing %s chunk %s: %s", data_type, chunk_id, e)
            raise

    def store_gradients(
        self,
        batch_idx: int,
        gradients: List[torch.Tensor],
        is_test: bool = False,
    ) -> None:
        """Store gradients directly in tensor format with async write.

        Args:
            batch_idx: Index of the batch to store gradients for.
            gradients: List of gradient tensors to store.
            is_test: Whether these are test gradients (currently skipped).

        Raises:
            RuntimeError: If failed to get new buffer after overflow.
        """
        if is_test:
            return  # Skip test gradients for now

        # Detect layer dimensions on first store
        if self.layer_dims is None:
            self.layer_dims = [g.shape[1] if g.numel() > 0 else 0 for g in gradients]
            self.total_proj_dim = sum(self.layer_dims)
            logger.debug(
                f"Detected layer dimensions: {len(self.layer_dims)} layers, total={self.total_proj_dim}",
            )

        chunk_id = self.get_chunk_id(batch_idx)
        buffer_key = ("gradients", chunk_id)
        pool = self._get_or_create_buffer_pool("gradients")

        with self._buffer_locks[buffer_key]:
            # Get or create buffer
            if buffer_key not in self._chunk_buffers:
                buffer = pool.get_buffer(timeout=5.0)
                if buffer is None:
                    logger.warning("Failed to get buffer from pool, creating new one")
                    buffer = ChunkBuffer(
                        tensor=torch.zeros(
                            self.max_samples_per_chunk,
                            self.total_proj_dim,
                            dtype=torch.float32,
                        ),
                        batch_indices=[],
                        batch_info=[],
                        current_row=0,
                        buffer_id=-1,
                    )
                self._chunk_buffers[buffer_key] = buffer

            buffer = self._chunk_buffers[buffer_key]

            # Concatenate gradients for this batch
            batch_size = next((g.shape[0] for g in gradients if g.numel() > 0), 0)
            if batch_size == 0:
                return

            # Build concatenated tensor for this batch
            batch_features = []
            for _layer_idx, (grad, dim) in enumerate(zip(gradients, self.layer_dims)):
                if grad.numel() > 0:
                    batch_features.append(grad.cpu())
                else:
                    # Zero padding for missing layers
                    batch_features.append(torch.zeros(batch_size, dim))

            batch_tensor = torch.cat(batch_features, dim=1)

            # Write to buffer
            start_row = buffer.current_row
            end_row = start_row + batch_size

            if end_row > self.max_samples_per_chunk:
                logger.warning(
                    "Chunk buffer overflow. Consider increasing max_samples_per_chunk",
                )
                # Force flush and start new buffer
                self._async_flush_chunk_buffer(buffer_key)

                # Get new buffer and retry
                buffer = pool.get_buffer(timeout=5.0)
                if buffer is None:
                    msg = "Failed to get new buffer after overflow"
                    raise RuntimeError(msg)
                self._chunk_buffers[buffer_key] = buffer
                start_row = 0
                end_row = batch_size

            # Direct copy to pre-allocated buffer
            buffer.tensor[start_row:end_row] = batch_tensor
            buffer.batch_indices.append(batch_idx)
            buffer.batch_info.append(
                {
                    "batch_idx": batch_idx,
                    "start_row": start_row,
                    "end_row": end_row,
                },
            )
            buffer.current_row = end_row

            # Check if chunk is complete
            if self._is_chunk_complete("gradients", chunk_id):
                self._async_flush_chunk_buffer(buffer_key)

    def _async_flush_chunk_buffer(self, buffer_key: Tuple[str, int]) -> None:
        """Asynchronously write chunk buffer to disk without blocking.

        Args:
            buffer_key: Tuple of (data_type, chunk_id) identifying the buffer to flush.
        """
        if buffer_key not in self._chunk_buffers:
            return

        buffer = self._chunk_buffers.pop(buffer_key)
        data_type, chunk_id = buffer_key

        # Clone the filled portion to a new tensor so we can return the buffer immediately
        filled_tensor = buffer.tensor[: buffer.current_row].clone().contiguous()
        batch_info_copy = buffer.batch_info.copy()

        # Return buffer to pool immediately after cloning
        pool = self._get_or_create_buffer_pool(data_type)
        pool.return_buffer(buffer)

        # Mark as pending write
        with self._pending_writes_lock:
            self._pending_writes.add((data_type, chunk_id))

        # Submit to write queue without the buffer (since we already returned it)
        write_task = (data_type, chunk_id, filled_tensor, batch_info_copy, None)

        try:
            self._write_queue.put_nowait(write_task)
        except queue.Full:
            logger.debug("Write queue full, blocking on write for chunk %s", chunk_id)
            self._write_queue.put(write_task)  # Block if queue is full

    def store_ifvp(self, batch_idx: int, ifvp: List[torch.Tensor]) -> None:
        """Store IFVP directly in tensor format with async write.

        Args:
            batch_idx: Index of the batch to store IFVP for.
            ifvp: List of IFVP tensors to store.

        Raises:
            RuntimeError: If failed to get new buffer after overflow.
        """
        self._ensure_layer_dims()

        chunk_id = self.get_chunk_id(batch_idx)
        buffer_key = ("ifvp", chunk_id)
        pool = self._get_or_create_buffer_pool("ifvp")

        with self._buffer_locks[buffer_key]:
            # Get or create buffer
            if buffer_key not in self._chunk_buffers:
                buffer = pool.get_buffer(timeout=5.0)
                if buffer is None:
                    logger.warning("Failed to get buffer from pool, creating new one")
                    buffer = ChunkBuffer(
                        tensor=torch.zeros(
                            self.max_samples_per_chunk,
                            self.total_proj_dim,
                            dtype=torch.float32,
                        ),
                        batch_indices=[],
                        batch_info=[],
                        current_row=0,
                        buffer_id=-1,
                    )
                self._chunk_buffers[buffer_key] = buffer

            buffer = self._chunk_buffers[buffer_key]

            # Concatenate IFVP for this batch
            batch_size = next((v.shape[0] for v in ifvp if v.numel() > 0), 0)
            if batch_size == 0:
                return

            # Build concatenated tensor
            batch_features = []
            for _layer_idx, (vec, dim) in enumerate(zip(ifvp, self.layer_dims)):
                if vec.numel() > 0:
                    batch_features.append(vec.cpu())
                else:
                    batch_features.append(torch.zeros(batch_size, dim))

            batch_tensor = torch.cat(batch_features, dim=1)

            # Write to buffer
            start_row = buffer.current_row
            end_row = start_row + batch_size

            if end_row > self.max_samples_per_chunk:
                self._async_flush_chunk_buffer(buffer_key)
                buffer = pool.get_buffer(timeout=5.0)
                if buffer is None:
                    msg = "Failed to get new buffer after overflow"
                    raise RuntimeError(msg)
                self._chunk_buffers[buffer_key] = buffer
                start_row = 0
                end_row = batch_size

            buffer.tensor[start_row:end_row] = batch_tensor
            buffer.batch_indices.append(batch_idx)
            buffer.batch_info.append(
                {
                    "batch_idx": batch_idx,
                    "start_row": start_row,
                    "end_row": end_row,
                },
            )
            buffer.current_row = end_row

            if self._is_chunk_complete("ifvp", chunk_id):
                self._async_flush_chunk_buffer(buffer_key)

    def _is_chunk_complete(self, data_type: str, chunk_id: int) -> bool:
        """Check if all batches in a chunk have been stored.

        Args:
            data_type: Type of data being checked.
            chunk_id: ID of the chunk to check.

        Returns:
            bool: True if all expected batches in the chunk have been stored.
        """
        if self.current_batch_range is None:
            return False

        start_batch, end_batch = self.current_batch_range
        chunk_start = chunk_id * self.chunk_size
        chunk_end = (chunk_id + 1) * self.chunk_size

        # Expected batches in this chunk
        expected_start = max(chunk_start, start_batch)
        expected_end = min(chunk_end, end_batch)

        if expected_start >= expected_end:
            return False

        # Check buffer
        buffer_key = (data_type, chunk_id)
        if buffer_key not in self._chunk_buffers:
            return False

        buffer = self._chunk_buffers[buffer_key]
        stored_batches = set(buffer.batch_indices)
        expected_batches = set(range(expected_start, expected_end))

        return expected_batches.issubset(stored_batches)

    def finalize_batch_range(self) -> None:
        """Flush any remaining buffers and wait for writes."""
        if self.current_batch_range is None:
            return

        # Flush all remaining buffers
        remaining_buffers = list(self._chunk_buffers.keys())
        for buffer_key in remaining_buffers:
            with self._buffer_locks[buffer_key]:
                self._async_flush_chunk_buffer(buffer_key)

        # Wait for all pending writes
        while True:
            with self._pending_writes_lock:
                if len(self._pending_writes) == 0:
                    break
            time.sleep(0.1)

        self.current_batch_range = None

    def _retrieve_batch_data_with_cache(
        self,
        data_type: str,
        batch_idx: int,
    ) -> List[torch.Tensor]:
        """Retrieve data with read caching.

        Args:
            data_type: Type of data to retrieve.
            batch_idx: Index of the batch to retrieve.

        Returns:
            List[torch.Tensor]: List of tensors split by layers.
        """
        self._ensure_layer_dims()

        # Check if in current buffer
        chunk_id = self.get_chunk_id(batch_idx)
        buffer_key = (data_type, chunk_id)

        # Check active buffer first
        with self._buffer_locks[buffer_key]:
            if buffer_key in self._chunk_buffers:
                buffer = self._chunk_buffers[buffer_key]
                # Find batch in buffer
                for info in buffer.batch_info:
                    if info["batch_idx"] == batch_idx:
                        start_row = info["start_row"]
                        end_row = info["end_row"]
                        batch_tensor = buffer.tensor[start_row:end_row]
                        return self._split_tensor_to_layers(batch_tensor)

        # Check read cache
        cache_key = (data_type, chunk_id)
        current_time = time.time()

        with self._read_cache_lock:
            if cache_key in self._read_cache:
                tensor, metadata, _ = self._read_cache[cache_key]
                self._read_cache[cache_key] = (
                    tensor,
                    metadata,
                    current_time,
                )  # Update access time

                # Find batch in cached chunk
                for info in metadata["batches"]:
                    if info["batch_idx"] == batch_idx:
                        start_row = info["start_row"]
                        end_row = info["end_row"]
                        batch_tensor = tensor[start_row:end_row]
                        return self._split_tensor_to_layers(batch_tensor)

        # Load from disk
        subdir = self._get_chunk_subdir(data_type)
        chunk_path = os.path.join(self.cache_dir, subdir)

        chunk_files = ChunkedMemoryMapHandler.find_chunk_files(chunk_path, data_type)

        for chunk_filename in chunk_files:
            metadata = ChunkedMemoryMapHandler.read_chunk_metadata(
                chunk_path,
                chunk_filename,
            )

            # Check if batch is in this chunk
            batch_found = False
            for info in metadata["batches"]:
                if info["batch_idx"] == batch_idx:
                    batch_found = True
                    break

            if batch_found:
                # Load entire chunk and cache it
                tensor, metadata = ChunkedMemoryMapHandler.load_chunk_tensor(
                    chunk_path,
                    chunk_filename,
                )

                # Update read cache
                with self._read_cache_lock:
                    self._read_cache[cache_key] = (tensor, metadata, current_time)

                    # Evict oldest entries if cache is full
                    if len(self._read_cache) > self._max_read_cache_size:
                        oldest_key = min(
                            self._read_cache.keys(),
                            key=lambda k: self._read_cache[k][2],
                        )
                        del self._read_cache[oldest_key]

                # Find and return batch data
                for info in metadata["batches"]:
                    if info["batch_idx"] == batch_idx:
                        start_row = info["start_row"]
                        end_row = info["end_row"]
                        batch_tensor = tensor[start_row:end_row]
                        return self._split_tensor_to_layers(batch_tensor)

        # Return empty tensors
        return [torch.tensor([]) for _ in range(len(self.layer_dims))]

    def retrieve_gradients(
        self,
        batch_idx: int,
        is_test: bool = False,
    ) -> List[torch.Tensor]:
        """Retrieve gradients for a batch and split into layers.

        Args:
            batch_idx: Index of the batch to retrieve gradients for.
            is_test: Whether to retrieve test gradients (currently unused).

        Returns:
            List[torch.Tensor]: List of gradient tensors split by layers.
        """
        return self._retrieve_batch_data_with_cache("gradients", batch_idx)

    def retrieve_ifvp(self, batch_idx: int) -> List[torch.Tensor]:
        """Retrieve IFVP for a batch and split into layers.

        Args:
            batch_idx: Index of the batch to retrieve IFVP for.

        Returns:
            List[torch.Tensor]: List of IFVP tensors split by layers.
        """
        return self._retrieve_batch_data_with_cache("ifvp", batch_idx)

    def _split_tensor_to_layers(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Split concatenated tensor back into per-layer tensors.

        Args:
            tensor: Concatenated tensor to split.

        Returns:
            List[torch.Tensor]: List of per-layer tensors.
        """
        result = []
        start_idx = 0
        for dim in self.layer_dims:
            end_idx = start_idx + dim
            result.append(tensor[:, start_idx:end_idx].contiguous())
            start_idx = end_idx

        return result

    def create_gradient_dataloader(
        self,
        data_type: str,
        batch_size: int = 1,
        pin_memory: bool = True,
        batch_range: Optional[Tuple[int, int]] = None,
    ) -> Optional[torch.utils.data.DataLoader]:
        """Create a DataLoader for loading chunked data with prefetching.

        Args:
            data_type: Type of data to load.
            batch_size: Batch size for the DataLoader.
            pin_memory: Whether to use pinned memory.
            batch_range: Optional range of batches to load.

        Returns:
            Optional[torch.utils.data.DataLoader]: DataLoader for the chunked data, or None if creation failed.
        """
        from .prefetch_dataset import create_tensor_dataloader

        return create_tensor_dataloader(
            disk_io=self,
            data_type=data_type,
            batch_size=batch_size,
            pin_memory=pin_memory,
            batch_range=batch_range,
            prefetch_factor=4,  # Prefetch 2 chunks ahead
        )

    def store_preconditioner(
        self,
        layer_idx: int,
        preconditioner: Optional[torch.Tensor],
    ) -> None:
        """Store a preconditioner for a layer.

        Args:
            layer_idx: Index of the layer to store preconditioner for.
            preconditioner: Preconditioner tensor to store, or None to skip.
        """
        if preconditioner is None:
            return

        cpu_precond = (
            preconditioner.cpu()
            if preconditioner.device.type != "cpu"
            else preconditioner
        )
        file_path = os.path.join(self.cache_dir, "precond", f"layer_{layer_idx}.pt")

        pathlib.Path(os.path.dirname(file_path)).mkdir(exist_ok=True, parents=True)

        future = self.executor.submit(torch.save, cpu_precond, file_path)
        self.futures.append(future)

    def retrieve_preconditioner(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Retrieve a preconditioner for a layer.

        Args:
            layer_idx: Index of the layer to retrieve preconditioner for.

        Returns:
            Optional[torch.Tensor]: Preconditioner tensor, or None if not found.
        """
        file_path = os.path.join(self.cache_dir, "precond", f"layer_{layer_idx}.pt")
        if pathlib.Path(file_path).exists():
            return torch.load(file_path, weights_only=True)
        return None

    def _get_chunk_subdir(self, data_type: str) -> str:
        """Get subdirectory for a data type.

        Args:
            data_type: Type of data.

        Returns:
            str: Subdirectory name for the data type.
        """
        return "grad" if data_type == "gradients" else data_type

    def wait_for_async_operations(self) -> None:
        """Wait for all async operations to complete."""
        if self._shutdown:
            return  # Skip if we're shutting down

        # Wait for write queue to empty
        while not self._write_queue.empty():
            time.sleep(0.1)

        # Wait for pending writes
        while True:
            with self._pending_writes_lock:
                if len(self._pending_writes) == 0:
                    break
            time.sleep(0.1)

        # Wait for other futures
        for future in self.futures:
            try:
                future.result()
            except Exception as e:
                logger.error("Async operation failed: %s", e)
        self.futures = []

    def has_preconditioners(self) -> bool:
        """Check if preconditioners exist.

        Returns:
            bool: True if preconditioner files exist in cache directory.
        """
        precond_dir = os.path.join(self.cache_dir, "precond")
        if not pathlib.Path(precond_dir).exists():
            return False
        return any(f.endswith(".pt") for f in os.listdir(precond_dir))

    def has_ifvp(self) -> bool:
        """Check if IFVP data exists.

        Returns:
            bool: True if IFVP chunk files exist in cache directory.
        """
        ifvp_dir = os.path.join(self.cache_dir, "ifvp")
        if not pathlib.Path(ifvp_dir).exists():
            return False
        return len(ChunkedMemoryMapHandler.find_chunk_files(ifvp_dir, "ifvp")) > 0

    def __del__(self) -> None:
        """Cleanup on destruction."""
        self._shutdown = True

        if hasattr(self, "_write_queue"):
            # Signal all write workers to stop
            for _ in self._write_workers:
                try:
                    self._write_queue.put(None)
                except:
                    pass

        if hasattr(self, "_write_workers"):
            # Wait for all workers to finish
            for worker in self._write_workers:
                worker.join(timeout=2.0)

        if hasattr(self, "executor"):
            # Flush remaining buffers
            try:
                remaining = list(self._chunk_buffers.keys())
                for buffer_key in remaining:
                    # Use synchronous flush for cleanup
                    if buffer_key in self._chunk_buffers:
                        buffer = self._chunk_buffers.pop(buffer_key)
                        data_type, chunk_id = buffer_key
                        filled_tensor = buffer.tensor[: buffer.current_row].clone()
                        self._write_chunk_tensor_sync(
                            data_type,
                            chunk_id,
                            filled_tensor,
                            buffer.batch_info,
                        )
            except Exception as e:
                logger.error("Error during cleanup: %s", e)

            self.wait_for_async_operations()
            self.executor.shutdown(wait=True)
