"""Memory-mapped file operations with non-blocking writes."""
from __future__ import annotations

import json
import logging
import os
import pathlib
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ChunkedMemoryMapHandler:
    """Optimized handler for memory-mapped file operations with non-blocking writes."""

    @staticmethod
    def write_chunk(
        save_path: str,
        data_type: str,
        tensor: torch.Tensor,
        batch_info: List[Dict[str, Any]],
        layer_dims: List[int],
        dtype: str = "float32",
        flush_mode: str = "async",  # 'async', 'sync', or 'none'
    ) -> str:
        """Write a tensor chunk to memory-mapped file with configurable flush behavior.

        Args:
            save_path: Directory to save files
            data_type: Type of data being stored (gradients, ifvp, etc.)
            tensor: Pre-concatenated tensor of shape (total_samples, total_proj_dim)
            batch_info: List of batch metadata dicts with keys: batch_idx, start_row, end_row
            layer_dims: List of projection dimensions for each layer
            dtype: NumPy data type to use for storage
            flush_mode: How to handle flushing ('async', 'sync', or 'none')

        Returns:
            str: The generated chunk filename (without extension)

        Raises:
            Exception: If there's an error writing the chunked memory-mapped file.
        """
        pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)

        # Generate filename based on batch range
        batch_indices = [info["batch_idx"] for info in batch_info]
        batch_start = min(batch_indices)
        batch_end = max(batch_indices)

        chunk_filename = f"chunk_{data_type}_{batch_start}_{batch_end}"
        mmap_path = os.path.join(save_path, f"{chunk_filename}.mmap")
        metadata_path = os.path.join(save_path, f"{chunk_filename}_metadata.json")

        # Create memory-mapped file
        try:
            # Determine storage dtype
            storage_dtype = "uint16" if tensor.dtype == torch.bfloat16 else dtype

            # Ensure tensor is on CPU and contiguous
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            tensor = tensor.contiguous()

            # Option 1: Direct file write without mmap for better async behavior
            if flush_mode == "async":
                # Convert to numpy array
                if tensor.dtype == torch.bfloat16:
                    np_array = tensor.view(torch.uint16).numpy()
                else:
                    np_array = tensor.numpy()

                # Save using numpy's memmap which doesn't require explicit flush
                np_memmap = np.memmap(
                    mmap_path,
                    dtype=np.dtype(storage_dtype),
                    mode="w+",
                    shape=tensor.shape,
                )
                np_memmap[:] = np_array

                # Let the OS handle flushing in the background
                # No explicit flush() call - this allows the write to be non-blocking
                del np_memmap  # This triggers implicit flush but doesn't block

            else:
                # Original behavior for compatibility
                mmap = np.memmap(
                    mmap_path,
                    dtype=np.dtype(storage_dtype),
                    mode="w+",
                    shape=tensor.shape,
                )

                # Write tensor data directly
                if tensor.dtype == torch.bfloat16:
                    uint16_view = tensor.view(torch.uint16)
                    mmap[:] = uint16_view.numpy()
                else:
                    mmap[:] = tensor.numpy()

                # Handle flush based on mode
                if flush_mode == "sync":
                    mmap.flush()  # Blocking flush
                elif flush_mode == "none":
                    pass  # No flush, let OS handle it

                del mmap  # Cleanup

            # Save metadata - this is small and fast
            metadata = {
                "chunk_filename": chunk_filename,
                "data_type": data_type,
                "batch_start": batch_start,
                "batch_end": batch_end,
                "storage_dtype": storage_dtype,
                "shape": list(tensor.shape),
                "layer_dims": layer_dims,
                "batches": batch_info,
                "flush_mode": flush_mode,  # Track how it was written
            }

            with pathlib.Path(metadata_path).open("w", encoding="utf-8") as f:
                json.dump(metadata, f, separators=(",", ":"))

            logger.debug(
                f"Wrote tensor chunk {chunk_filename}: shape={tensor.shape}, flush_mode={flush_mode}",
            )

            return chunk_filename

        except Exception as e:
            logger.exception(f"Error writing chunked memory-mapped file {mmap_path}: {e!s}")
            raise

    @staticmethod
    @contextmanager
    def read_chunk(path: str, chunk_filename: str, force_load: bool = False):
        """Context manager to read a chunked memory-mapped file as tensor.

        Args:
            path: Directory containing the memory-mapped file
            chunk_filename: Name of the chunk file
            force_load: If True, load entire file into memory for better performance

        Yields:
            Tuple of (tensor, metadata) where tensor has shape (total_samples, total_proj_dim)
        """
        mmap_path = os.path.join(path, f"{chunk_filename}.mmap")
        metadata_path = os.path.join(path, f"{chunk_filename}_metadata.json")

        mmap = None
        try:
            # Load metadata
            with pathlib.Path(metadata_path).open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            storage_dtype = metadata["storage_dtype"]
            shape = tuple(metadata["shape"])

            if force_load:
                # Load entire file into memory for better performance
                # This is useful when we know we'll access all data
                data = np.fromfile(mmap_path, dtype=np.dtype(storage_dtype))
                data = data.reshape(shape)

                if storage_dtype == "uint16":
                    tensor = torch.from_numpy(data.astype(np.int16)).view(
                        torch.bfloat16,
                    )
                else:
                    tensor = torch.from_numpy(data)

                yield tensor, metadata
            else:
                # Use memory mapping for random access
                # Open with copy-on-write to avoid modifying the file
                mmap = np.memmap(
                    mmap_path,
                    dtype=np.dtype(storage_dtype),
                    mode="c",
                    shape=shape,
                )

                if storage_dtype == "uint16":
                    # For bfloat16, we need special handling
                    tensor = torch.as_tensor(mmap, dtype=torch.int16).view(
                        torch.bfloat16,
                    )
                else:
                    # Directly create tensor view of mmap - no copy!
                    tensor = torch.as_tensor(mmap)

                yield tensor, metadata

        finally:
            if mmap is not None:
                del mmap

    @staticmethod
    def read_chunk_metadata(path: str, chunk_filename: str) -> Dict[str, Any]:
        """Read metadata from a chunked file.

        Args:
            path: Directory containing the metadata file.
            chunk_filename: Name of the chunk file (without extension).

        Returns:
            Dict[str, Any]: Dictionary containing the chunk metadata.
        """
        metadata_path = os.path.join(path, f"{chunk_filename}_metadata.json")
        with pathlib.Path(metadata_path).open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_chunk_tensor(
        path: str,
        chunk_filename: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load entire chunk as tensor with metadata.
        Uses force_load for better performance when loading entire chunks.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file

        Returns:
            Tuple of (tensor, metadata)
        """
        with ChunkedMemoryMapHandler.read_chunk(
            path,
            chunk_filename,
            force_load=True,
        ) as (tensor, metadata):
            # Return the tensor directly since force_load already made a copy
            return tensor, metadata

    @staticmethod
    def load_batch_slice(
        path: str,
        chunk_filename: str,
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        """Load a specific batch slice from a chunk.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file
            batch_idx: Batch index to load

        Returns:
            Tensor slice for the batch or None if not found
        """
        with ChunkedMemoryMapHandler.read_chunk(path, chunk_filename) as (
            tensor,
            metadata,
        ):
            # Find the batch in metadata
            for batch_info in metadata["batches"]:
                if batch_info["batch_idx"] == batch_idx:
                    start_row = batch_info["start_row"]
                    end_row = batch_info["end_row"]
                    return tensor[start_row:end_row].clone()

            return None

    @staticmethod
    def load_chunk_batch_range(
        path: str,
        chunk_filename: str,
        batch_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[int, int]]]:
        """Load chunk with optional batch filtering.
        Uses force_load=True for better performance.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file
            batch_range: Optional (start, end) range to filter batches

        Returns:
            Tuple of:
                - Tensor of shape (filtered_samples, total_proj_dim)
                - Mapping from batch_idx to (start_row, end_row) in the returned tensor
        """
        with ChunkedMemoryMapHandler.read_chunk(
            path,
            chunk_filename,
            force_load=True,
        ) as (tensor, metadata):
            if batch_range is None:
                batch_mapping = {
                    info["batch_idx"]: (info["start_row"], info["end_row"])
                    for info in metadata["batches"]
                }
                return tensor, batch_mapping

            # Filter batches
            start_batch, end_batch = batch_range
            valid_rows = []
            new_mapping = {}
            new_offset = 0

            for batch_info in metadata["batches"]:
                batch_idx = batch_info["batch_idx"]
                if start_batch <= batch_idx < end_batch:
                    start_row = batch_info["start_row"]
                    end_row = batch_info["end_row"]
                    batch_size = end_row - start_row

                    valid_rows.extend(range(start_row, end_row))
                    new_mapping[batch_idx] = (new_offset, new_offset + batch_size)
                    new_offset += batch_size

            if valid_rows:
                # Since we used force_load, tensor is already in memory
                filtered_tensor = tensor[valid_rows]
                return filtered_tensor, new_mapping
            return torch.empty(0, tensor.shape[1]), {}

    @staticmethod
    def find_chunk_files(path: str, data_type: str) -> List[str]:
        """Find all chunk files for a specific data type.

        Args:
            path: Directory to search for chunk files.
            data_type: Type of data to search for (gradients, ifvp, etc.).

        Returns:
            List[str]: List of chunk filenames sorted by batch start index.
        """
        if not pathlib.Path(path).exists():
            return []

        chunk_files = []
        for filename in os.listdir(path):
            if (
                filename.endswith("_metadata.json")
                and f"chunk_{data_type}_" in filename
            ):
                chunk_name = filename.replace("_metadata.json", "")
                chunk_files.append(chunk_name)

        # Sort by batch start index
        def extract_batch_start(chunk_name):
            try:
                parts = chunk_name.split("_")
                if len(parts) >= 4:
                    return int(parts[2])
                return 0
            except (ValueError, IndexError):
                return 0

        return sorted(chunk_files, key=extract_batch_start)

    @staticmethod
    def ensure_chunk_written(
        path: str,
        chunk_filename: str,
        timeout: float = 60.0,
    ) -> bool:
        """Ensure a chunk has been fully written to disk.
        This can be used when you need to guarantee data persistence.

        Args:
            path: Directory containing the files
            chunk_filename: Name of the chunk file
            timeout: Maximum time to wait in seconds

        Returns:
            True if chunk is verified on disk, False if timeout
        """
        import time

        mmap_path = os.path.join(path, f"{chunk_filename}.mmap")
        metadata_path = os.path.join(path, f"{chunk_filename}_metadata.json")

        start_time = time.time()

        # First check if files exist
        while time.time() - start_time < timeout:
            if (
                pathlib.Path(mmap_path).exists()
                and pathlib.Path(metadata_path).exists()
            ):
                try:
                    # Try to load metadata to ensure it's complete
                    with pathlib.Path(metadata_path).open("r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Check if the file size matches expected size
                    expected_size = (
                        np.prod(metadata["shape"])
                        * np.dtype(metadata["storage_dtype"]).itemsize
                    )
                    actual_size = pathlib.Path(mmap_path).stat().st_size

                    if actual_size >= expected_size:
                        return True

                except Exception:
                    pass

            time.sleep(0.1)

        return False
