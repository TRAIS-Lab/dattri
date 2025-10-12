"""Simple metadata management with master worker approach.
Just restore the original metadata.py with one small addition.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import pathlib
import threading
import time
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manager for batch metadata with master worker coordination."""

    def __init__(self, cache_dir: str, layer_names: List[str]) -> None:
        """Initialize the metadata manager.

        Args:
            cache_dir: Directory for metadata files
            layer_names: Names of model layers
        """
        self.cache_dir = cache_dir
        self.layer_names = layer_names
        self.batch_info = {}  # Maps batch_idx -> {sample_count, start_idx}
        self.total_samples = 0
        self.layer_dims = None  # Store layer dimensions
        self.total_proj_dim = None  # Total projection dimension
        self._metadata_lock = threading.Lock()
        self._pending_batches = {}  # Buffer for batches before bulk save
        self._last_save_time = 0
        self._save_interval = 5.0  # Save every 5 seconds max

        if cache_dir:
            pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)
            self._load_metadata_if_exists()

        logger.debug(f"Initialized MetadataManager with {len(layer_names)} layers")

    def initialize_complete_dataset(
        self,
        train_dataloader,
        is_master_worker: bool = False,
    ) -> None:
        """Initialize complete dataset metadata. Only master worker (worker 0) does this.

        Args:
            train_dataloader: The training dataloader
            is_master_worker: True if this is worker 0, False otherwise

        Raises:
            RuntimeError: If master worker failed to initialize metadata within 30 seconds.
        """
        if not is_master_worker:
            # Non-master workers just load existing metadata
            self._load_metadata_if_exists()
            if not self.batch_info:
                logger.info(
                    "Non-master worker waiting for master to initialize metadata...",
                )
                # Wait for master worker to create metadata
                for _ in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    self._load_metadata_if_exists()
                    if self.batch_info:
                        logger.info("Metadata initialized by master worker")
                        break
                else:
                    msg = (
                        "Master worker failed to initialize metadata within 30 seconds"
                    )
                    raise RuntimeError(
                        msg,
                    )
            return

        # Only master worker (worker 0) initializes the complete dataset
        logger.info("Master worker initializing complete dataset metadata...")

        total_batches = len(train_dataloader)
        batch_size = train_dataloader.batch_size

        # Check if already initialized
        if len(self.batch_info) == total_batches:
            logger.info("Dataset metadata already complete, skipping initialization")
            return

        # Compute complete batch structure
        if hasattr(train_dataloader.dataset, "__len__"):
            dataset_size = len(train_dataloader.sampler)
            current_sample_idx = 0

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                # Handle last batch which might be smaller
                if batch_idx == total_batches - 1 and not getattr(
                    train_dataloader,
                    "drop_last",
                    False,
                ):
                    actual_batch_size = dataset_size - start_idx
                else:
                    actual_batch_size = min(batch_size, dataset_size - start_idx)

                self.batch_info[batch_idx] = {
                    "sample_count": actual_batch_size,
                    "start_idx": current_sample_idx,
                }
                current_sample_idx += actual_batch_size
        else:
            # Fallback: assume uniform batch sizes
            logger.warning(
                "Dataset doesn't support __len__, using uniform batch size assumption",
            )
            current_sample_idx = 0
            for batch_idx in range(total_batches):
                self.batch_info[batch_idx] = {
                    "sample_count": batch_size,
                    "start_idx": current_sample_idx,
                }
                current_sample_idx += batch_size

        self.total_samples = current_sample_idx

        # Save the complete metadata immediately
        self.save_metadata()

        logger.info(
            "Master worker initialized complete dataset: %s batches, %s samples",
            total_batches,
            current_sample_idx,
        )

    def add_batch_info(self, batch_idx: int, sample_count: int) -> None:
        """Add information about a batch with optimized batching.
        For non-master workers, this just validates but doesn't save metadata.

        Args:
            batch_idx: Index of the batch
            sample_count: Number of samples in the batch
        """
        with self._metadata_lock:
            if batch_idx in self.batch_info:
                # Validate sample count matches expected
                expected_count = self.batch_info[batch_idx]["sample_count"]
                if sample_count != expected_count:
                    logger.warning(
                        "Batch %s sample count mismatch. Expected: %s, Got: %s",
                        batch_idx,
                        expected_count,
                        sample_count,
                    )
            else:
                # This shouldn't happen if master worker initialized properly
                logger.warning(
                    "Batch %s not found in pre-initialized metadata",
                    batch_idx,
                )
                self._pending_batches[batch_idx] = {
                    "sample_count": sample_count,
                    "start_idx": self.total_samples,
                }
                self.total_samples += sample_count

    def set_layer_dims(self, layer_dims: List[int]) -> None:
        """Set the layer dimensions.

        Args:
            layer_dims: List of dimensions for each layer
        """
        with self._metadata_lock:
            self.layer_dims = layer_dims
            self.total_proj_dim = sum(layer_dims) if layer_dims else None
            logger.debug(
                f"Set layer dimensions: {len(layer_dims)} layers, total={self.total_proj_dim}",
            )

    def _flush_pending_batches(self) -> None:
        """Flush pending batches to the main batch_info dict."""
        if not self._pending_batches:
            return

        # Merge pending batches
        self.batch_info.update(self._pending_batches)
        self._pending_batches.clear()
        self._last_save_time = time.time()

    def get_total_samples(self) -> int:
        """Get the total number of samples across all batches.

        Returns:
            Total number of samples
        """
        return self.total_samples

    def get_batch_to_sample_mapping(self) -> Dict[int, Tuple[int, int]]:
        """Get mapping from batch indices to sample index ranges.

        Returns:
            Dictionary mapping batch indices to (start_idx, end_idx) tuples
        """
        # Ensure pending batches are flushed
        with self._metadata_lock:
            self._flush_pending_batches()

        return {
            batch_idx: (info["start_idx"], info["start_idx"] + info["sample_count"])
            for batch_idx, info in self.batch_info.items()
        }

    def get_total_batches(self) -> int:
        """Get total number of batches in the dataset.

        Returns:
            Total number of batches
        """
        if not self.batch_info:
            return 0

        # Calculate total batches from batch indices (same as original logic)
        batch_indices = sorted(self.batch_info.keys())
        if not batch_indices:
            return 0

        max_batch_idx = max(batch_indices)
        return max_batch_idx + 1

    def save_metadata(self) -> None:
        """Save metadata to disk."""
        if not self.cache_dir:
            return

        with self._metadata_lock:
            # Flush any pending batches first
            self._flush_pending_batches()

        metadata_path = self._get_metadata_path()
        temp_path = metadata_path + ".tmp"

        try:
            # Recompute start indices to ensure consistency
            sorted_batches = sorted(self.batch_info.keys())
            current_idx = 0
            batch_info_corrected = {}

            for batch_idx in sorted_batches:
                sample_count = self.batch_info[batch_idx]["sample_count"]
                batch_info_corrected[batch_idx] = {
                    "start_idx": current_idx,
                    "sample_count": sample_count,
                }
                current_idx += sample_count

            # Prepare serializable format (convert int keys to strings for JSON)
            serializable_info = {
                str(idx): info for idx, info in batch_info_corrected.items()
            }

            metadata = {
                "batch_info": serializable_info,
                "layer_names": self.layer_names,
                "layer_dims": self.layer_dims,
                "total_proj_dim": self.total_proj_dim,
                "total_samples": current_idx,
                "timestamp": time.time(),
            }

            # Write to temporary file first (atomic operation)
            with pathlib.Path(temp_path).open("w", encoding="utf-8") as f:
                json.dump(metadata, f, separators=(",", ":"))  # Compact format

            # Atomic rename
            pathlib.Path(temp_path).replace(metadata_path)

            # Update our internal state to match what we saved
            with self._metadata_lock:
                self.batch_info = batch_info_corrected
                self.total_samples = current_idx

            logger.debug(f"Saved metadata for {len(self.batch_info)} batches")

        except Exception as e:
            logger.exception("Error saving metadata: %s", e)
            # Clean up temp file if it exists
            if pathlib.Path(temp_path).exists():
                with contextlib.suppress(builtins.BaseException):
                    pathlib.Path(temp_path).unlink()

    def _get_metadata_path(self) -> str:
        """Get the path to the metadata file.

        Returns:
            str: The path to the batch_metadata.json file.
        """
        return os.path.join(self.cache_dir, "batch_metadata.json")

    def _load_metadata_if_exists(self) -> None:
        """Load metadata from disk if it exists."""
        metadata_path = self._get_metadata_path()
        if pathlib.Path(metadata_path).exists():
            try:
                with pathlib.Path(metadata_path).open("r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Convert string keys back to integers for batch info
                self.batch_info = {
                    int(batch_idx): {
                        "sample_count": info["sample_count"],
                        "start_idx": info["start_idx"],
                    }
                    for batch_idx, info in metadata["batch_info"].items()
                }

                self.total_samples = metadata.get("total_samples", 0)

                # Load layer information
                if "layer_names" in metadata:
                    self.layer_names = metadata["layer_names"]

                if "layer_dims" in metadata:
                    self.layer_dims = metadata["layer_dims"]
                    self.total_proj_dim = metadata.get("total_proj_dim")
                    logger.debug("Loaded layer dimensions from metadata")

                logger.info(f"Loaded metadata for {len(self.batch_info)} batches")

            except Exception as e:
                logger.exception("Error loading metadata: %s", e)

    def __del__(self) -> None:
        """Ensure metadata is saved on destruction."""
        try:
            if hasattr(self, "_pending_batches") and self._pending_batches:
                logger.info("Saving pending metadata on destruction")
                self.save_metadata()
        except Exception as e:
            logger.exception("Error saving metadata during cleanup: %s", e)
